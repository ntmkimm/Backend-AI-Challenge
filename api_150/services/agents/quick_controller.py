from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from langchain_core.messages import HumanMessage, AIMessage

from services.redis_service import RedisService
from .quick_agentsgraph import build_quick_agent, QuickInSchema, QuickOutSchema
@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime
@dataclass
class AgentResponse:
    message: str
    tool_outputs: Optional[List[Dict]] = None
class QuickAgentController:
    def __init__(self, provider: str, api_key: str,
                 model: Optional[str], redis_service: Optional[RedisService]):
        self.redis = redis_service
        self.app = build_quick_agent(provider, api_key, model, use_memory=False)

    async def process_message(
        self,
        user_message: str,
        conversation_history: Optional[List[ChatMessage]] = None,
        user_id: str = "anonymous"
    ) -> AgentResponse:
        msgs = []
        if conversation_history:
            for m in conversation_history[-5:]:
                if m.role == "user":
                    msgs.append(HumanMessage(content=m.content))
                elif m.role == "assistant":
                    msgs.append(AIMessage(content=m.content))
        msgs.append(HumanMessage(content=user_message))

        state: QuickOutSchema = await self.app.ainvoke({"messages": msgs},
            config={"configurable": {"thread_id": f"{user_id}:quick_chat"}})

        messages = state["messages"]
        tool_outputs = state.get("tool_outputs", [])

        final_text = ""
        
        # --- Start of Modification ---
        
        # First, check if the last action resulted in a direct tool output we should return.
        # This is for when we want the raw tool data without LLM summarization.
        if not final_text and tool_outputs:
            last_tool_output = tool_outputs[-1]
            # If the tool call was successful and it was a query refinement,
            # format its output as the definitive response.
            if last_tool_output.get('ok') and last_tool_output.get('tool') == 'query_refinement_tool':
                # Serialize the 'output' part of the tool run into a JSON string.
                # The 'output' key comes from the `ToolRun` schema.
                output_content = last_tool_output.get('output', [])
                final_text = json.dumps({"refined_queries": output_content})

        # If we didn't get text from the tool output, fall back to the original logic
        # of finding the last conversational message from the AI.
        if not final_text:
            for m in reversed(messages):
                if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                    final_text = m.content
                    break
        
        # --- End of Modification ---

        if self.redis and conversation_history is not None:
            try:
                # Caching logic can be added here if needed
                pass
            except Exception as e:
                print(f"Redis caching error: {e}")

        return AgentResponse(message=final_text, tool_outputs=tool_outputs)