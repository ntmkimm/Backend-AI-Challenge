# services/agents/quick_agent_graph.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
import asyncio
import json
# Add SystemMessage to the import list
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from services.agents.llms import make_llm
from services.agents.tools import (
    duckduckgo_web_search_tool,
    duckduckgo_image_search_tool,
    make_query_refinement_tool,
)

# ---------- Schemas ----------

class QuickInSchema(TypedDict):
    """Input to the quick agent graph."""
    messages: List[BaseMessage]

class ToolRun(TypedDict, total=False):
    tool: str
    ok: bool
    output: Any
    error: str
    meta: Dict[str, Any] # Added meta to capture extra info from tool output

class QuickOutSchema(TypedDict):
    """Output of the quick agent graph."""
    messages: List[BaseMessage]
    tool_outputs: List[ToolRun]

# ---------- Builder ----------

def build_quick_agent(
    provider: str,
    api_key: str,
    model: Optional[str] = None,
    *,
    use_memory: bool = False,
):
    """
    Build a simplified agent for quick queries that only handles:
    - Web search (DuckDuckGo)
    - Image search (DuckDuckGo)
    - Query refinement
    
    No video search, no intent parsing, no pre-routing.
    """
    llm = make_llm(provider, api_key, model)
    query_refinement_for_model = make_query_refinement_tool(llm)

    # Only web-related tools
    tools = [
        duckduckgo_image_search_tool,
        duckduckgo_web_search_tool,
        query_refinement_for_model,
    ]
    
    model_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    # --- Start of Integration ---
    # 1. Define the guiding system prompt for the agent.
    system_prompt = SystemMessage(content="""\
You are a fast, minimalist search assistant orchestrating three tools:
- query_refinement_tool
- duckduckgo_web_search_tool
- duckduckgo_image_search_tool

## Core policy (VERY IMPORTANT)
- If the user does NOT explicitly ask to **search the web** or **find images**, your default and only action is to call **query_refinement_tool**.
- Only call **duckduckgo_web_search_tool** when the user expressly requests a web search (e.g., "search the web", "look this up online", "google/duckduckgo this", "find articles/news").
- Only call **duckduckgo_image_search_tool** when the user expressly requests images (e.g., "find images/pics/photos", "show me pictures", "image search").
- Do NOT call both web and image search in the same turn unless the user clearly asks for both.

## Refinement strategy (when you call query_refinement_tool)
You MUST choose the `refinement_strategy` param:
1) 'llm_only'  — Use for general queries with no proper nouns (no specific named people, places, orgs, events).
2) 'web_enriched' — Use ONLY if the query contains proper nouns or specific events (e.g., "Taylor Swift", "Eiffel Tower", "Super Bowl LVIII"), because these benefit from up-to-date context before rewriting.

### Required behavior
1) Analyze the last user message.
2) If the message contains an explicit command to search images → call duckduckgo_image_search_tool(query=...).
3) Else if the message contains an explicit command to search the web → call duckduckgo_web_search_tool(query=...).
4) Else → call query_refinement_tool with the correct `refinement_strategy`:
   - Example (general): 
     query_refinement_tool(
       initial_queries=[{"text":"a man walking his dog"}],
       refinement_strategy="llm_only"
     )
   - Example (proper nouns):
     query_refinement_tool(
       initial_queries=[{"text":"videos of the last Daft Punk concert"}],
       refinement_strategy="web_enriched"
     )

5) After any tool returns, produce a concise, actionable follow-up. 
   - If you only refined (no explicit search requested), briefly present the refined queries and say you can run a web or image search if they want.
   - If you performed a search, summarize top findings succinctly.

### Parsing hints
Treat these phrases as explicit image-search intent:
- "find images", "image search", "show me pictures/photos", "pics", "gallery"

Treat these phrases as explicit web-search intent:
- "search the web", "look this up online", "duckduckgo this", "google this", "find articles/news", "web results"

### Examples

# Example A (no explicit command → refinement only; general)
User: "a girl in a red ao dai near a 'Bún bò' sign"
Your tool call:
query_refinement_tool(
  initial_queries=[{"text":"a girl in a red ao dai near a 'Bún bò' sign"}],
  refinement_strategy="llm_only"
)

# Example B (no explicit command → refinement only; proper nouns)
User: "find clips from Super Bowl LVIII halftime show"
Your tool call:
query_refinement_tool(
  initial_queries=[{"text":"clips from Super Bowl LVIII halftime show"}],
  refinement_strategy="web_enriched"
)

# Example C (explicit image command)
User: "find images of Hanoi Old Quarter at night"
Your tool call:
duckduckgo_image_search_tool(query="Hanoi Old Quarter at night")

# Example D (explicit web command)
User: "search the web for latest research on CLIP variants"
Your tool call:
duckduckgo_web_search_tool(query="latest research on CLIP variants")

# Example E (both explicitly requested)
User: "search the web and also show images of Notre-Dame restoration progress"
Your tool calls, in order:
duckduckgo_web_search_tool(query="Notre-Dame restoration progress")
duckduckgo_image_search_tool(query="Notre-Dame restoration progress")

Follow these rules exactly. Default to refinement unless the user clearly commands a web or image search.
""")



    def call_model(state: QuickOutSchema) -> Dict[str, Any]:
        """
        Simple LLM call with tools.
        The LLM decides which tool to use based on the user's message.
        """
        # 2. Prepend the system prompt to the conversation history.
        #    This ensures the LLM always sees the instructions first.
        messages_with_prompt = [system_prompt] + state["messages"]
        
        response: AIMessage = model_with_tools.invoke(messages_with_prompt)
        return {"messages": state["messages"] + [response]}

    async def call_tools(state: QuickOutSchema) -> Dict[str, Any]:
        """Execute any tools called by the LLM."""
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {}

        tool_outputs: List[ToolRun] = list(state.get("tool_outputs", []))

        async def _run_one(tc):
            name = tc["name"]
            args = tc.get("args") or {}
            tool = tool_map.get(name)
            
            if tool is None:
                tool_outputs.append({"tool": name, "ok": False, "error": "not_found"})
                return ToolMessage(content=f"Tool '{name}' not found.", tool_call_id=tc["id"])

            try:
                # Tools now return a standardized dictionary from make_tool_output
                out = await tool.ainvoke(args)

                # The 'items' key holds the full list of results for the UI
                ui_payload = out.get("items", [])
                
                # A smaller slice of the results is fed back to the LLM
                agent_payload = ui_payload[:5]

                # Record the full tool output for the UI, including metadata
                tool_outputs.append({
                    "tool": name,
                    "ok": True,
                    "output": ui_payload,
                    "meta": out.get("meta", {})
                })

                # The ToolMessage for the LLM contains the smaller, summarized payload
                agent_msg_content = json.dumps({"tool": name, "items": agent_payload})
                return ToolMessage(content=agent_msg_content, tool_call_id=tc["id"])

            except Exception as e:
                tool_outputs.append({"tool": name, "ok": False, "error": str(e)})
                return ToolMessage(content=f"[tool_error] {e}", tool_call_id=tc["id"])

        tool_msgs = await asyncio.gather(*[_run_one(tc) for tc in last.tool_calls])

        return {
            "messages": state["messages"] + list(tool_msgs),
            "tool_outputs": tool_outputs,
        }

    # Build simple graph
    graph = StateGraph(
        state_schema=QuickOutSchema,
        input_schema=QuickInSchema,
        output_schema=QuickOutSchema,
    )

    # Add nodes
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)

    # Simple routing
    def should_call_tools(state: QuickOutSchema) -> str:
        """Check if the model wants to call tools."""
        last = state["messages"][-1] if state.get("messages") else None
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    # Connect nodes
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", should_call_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")  # After tools, go back to model for synthesis

    # Compile
    if use_memory:
        memory = MemorySaver()
        app = graph.compile(checkpointer=memory)
        output_path = "quick_agent_graph_memory.png"
        app.get_graph().draw_png(output_path)
        return app
    
    app = graph.compile()
    output_path = "quick_agent_graph.png"
    app.get_graph().draw_png(output_path)
    return app