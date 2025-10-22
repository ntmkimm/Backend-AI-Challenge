# services/agents/agent_graph.py
from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Optional
import asyncio
import json
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
# If you want checkpointing, uncomment MemorySaver and see controller note.
from langgraph.checkpoint.memory import MemorySaver
from services.agents.llms import make_llm
from services.agents.tools import (
    video_search_tool,
    chain_search_tool,
    frame_information_tool,
    history_get_tool,
    duckduckgo_web_search_tool,
    duckduckgo_image_search_tool,
    make_query_refinement_tool,
)
from .chain import build_chain
from .schema import LCAgentJSON
from langchain_core.prompts.chat import ChatPromptTemplate
# ---------- Strong schemas ----------

class InSchema(TypedDict):
    """Input to the graph."""
    messages: List[BaseMessage]          # history + new HumanMessage

class ToolRun(TypedDict, total=False):
    tool: str
    ok: bool
    output: Any
    error: str

class OutSchema(TypedDict):
    """Final output of the graph."""
    messages: List[BaseMessage]
    tool_outputs: List[ToolRun]
    structured_query: Optional[LCAgentJSON]
    routing_decision: Optional[str] # Add this line to store the router's choice
# ---------- Builder ----------

def build_tool_agent(
    provider: str,
    api_key: str,
    model: Optional[str] = None,
    *,
    use_memory: bool = False,   # set True if you later add a checkpointer
):
    llm = make_llm(provider, api_key, model)
    query_refinement_for_model = make_query_refinement_tool(llm)

    tools = [
        video_search_tool,
        chain_search_tool,
        frame_information_tool,
        history_get_tool,   
        duckduckgo_image_search_tool,
        duckduckgo_web_search_tool,
        query_refinement_for_model,  # ✅ Gemini-safe
    ]
    model_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    # Nodes must accept the current state (TypedDict) and return a dict of updates.
    def pre_router(state: InSchema) -> Dict[str, Any]:
        """
        Classifies the user's intent to either perform a video search
        or a direct tool call (like web search or refinement).
        """
        user_input = state["messages"][-1].content
        
        # Simple prompt for classification
        routing_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a routing agent. Your task is to classify the user's request.

                    - If the user is asking to find, search for, or see a video based on its content, respond with 'video_search'.
                    - If the user is asking to search the web, use an image search, or asking to refine a previous query, respond with 'direct_tool_use'.

                    Respond with ONLY 'video_search' or 'direct_tool_use'.""",
                ),
                ("human", "{input}"),
            ]
        )
        
        # Use a simple chain for the routing decision
        router_chain = routing_prompt | llm
        decision = router_chain.invoke({"input": user_input}).content
        
        # Clean up the response to be safe
        if "video_search" in decision:
            return {"routing_decision": "video_search"}
        else:
            return {"routing_decision": "direct_tool_use"}
    def intent_parser(state: InSchema) -> Dict[str, Any]:
        """Calls the initial chain to parse the user's intent."""
        chain = build_chain(llm)
        # Assuming the last message is the user's input
        # agent_graph.py → inside intent_parser
        MAX_PAIRS = 2
        def _tail_pairs(msgs):
            # take last 2 user/assistant turns (4 msgs max)
            out = []
            cnt = 0
            for m in reversed(msgs):
                if getattr(m, "type", None) == "human":
                    cnt += 1
                out.append(m)
                if cnt >= MAX_PAIRS:
                    break
            return list(reversed(out))
        
        user_input = state["messages"][-1].content
        history = _tail_pairs(state["messages"][:-1])


        # Invoke the chain
        structured_query: LCAgentJSON = chain.invoke({"history": history, "input": user_input})

        # Create an AIMessage from the chain's output
        ai_message = AIMessage(content=structured_query.message)

        return {
            "messages": state["messages"] + [ai_message],
            "structured_query": structured_query
        }

    def call_model(state: OutSchema) -> Dict[str, Any]:
        """One LLM step; may contain tool_calls."""
        response: AIMessage = model_with_tools.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    async def call_tools(state: OutSchema) -> Dict[str, Any]:
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
                out = await tool.ainvoke(args)

                # --- Dual payload handling ---
                agent_payload = out
                ui_payload = out

                # Preferred shape from tools: {"agent_view": [...], "full": [...]}
                if isinstance(out, dict) and "agent_view" in out and "full" in out:
                    agent_payload = out["agent_view"]
                    ui_payload = out["full"]
                # Backward-compat: if a list comes back, slice for agent, keep full for UI
                elif isinstance(out, list):
                    agent_payload = out[:5]
                    ui_payload = out

                # Record FULL payload for the UI
                tool_outputs.append({"tool": name, "ok": True, "output": ui_payload})

                # Feed ONLY the small slice to the LLM
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
    # Build graph with strong schemas (constructor arguments, like your example)
    # Build graph with strong schemas
    graph = StateGraph(
        state_schema=OutSchema,
        input_schema=InSchema,
        output_schema=OutSchema,
    )

    # Add all nodes to the graph
    graph.add_node("pre_router", pre_router) # Add the new router node
    graph.add_node("intent_parser", intent_parser)
    graph.add_node("model", call_model)
    graph.add_node("tools", call_tools)

    # --- Define the new routing logic ---

    def route_after_pre_router(state: OutSchema) -> str:
        """Decides where to go after the initial routing."""
        if state.get("routing_decision") == "video_search":
            return "intent_parser"
        return "model" # For 'direct_tool_use'

    def route_after_intent(state: OutSchema) -> str:
        """Determines the next step based on the chain's output."""
        if state.get("structured_query") and state["structured_query"].should_search:
            return "model"
        return END

    def should_call_tools(state: OutSchema) -> str:
        last = state["messages"][-1] if state.get("messages") else None
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END

    # --- Connect the graph edges ---

    graph.add_edge(START, "pre_router") # The new entry point
    
    # Conditional edge from the pre-router
    graph.add_conditional_edges(
        "pre_router",
        route_after_pre_router,
        {
            "intent_parser": "intent_parser",
            "model": "model"
        }
    )
    
    # The rest of the graph connections
    graph.add_conditional_edges("intent_parser", route_after_intent, {"model": "model", END: END})
    graph.add_conditional_edges("model", should_call_tools, {"tools": "tools", END: END})
    graph.add_edge("tools", "model")

    # If you enable checkpointing:
    if use_memory:
        memory = MemorySaver()
        app = graph.compile(checkpointer=memory)
        output_path="agent_graph_memory.png"
        app.get_graph().draw_png(output_path)
        return app
    app=graph.compile()
    output_path="agent_graph.png"
    app.get_graph().draw_png(output_path)
    return app