# services/agents/controller.py (only the changed parts)
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from langchain_core.messages import HumanMessage, AIMessage

from services.redis_service import RedisService
from .agentsgraph import build_tool_agent, InSchema, OutSchema

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime

@dataclass
class AgentResponse:
    message: str
    tool_outputs: Optional[List[Dict]] = None

class AgentController:
    def __init__(self, provider: str, api_key: str,
                 model: Optional[str], redis_service: Optional[RedisService]):
        self.redis = redis_service
        self.app = build_tool_agent(provider, api_key, model, use_memory=False)  # set True to enable memory

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

        # If you compiled with MemorySaver, include thread_id:
        state: OutSchema = await self.app.ainvoke({"messages": msgs},
            config={"configurable": {"thread_id": f"{user_id}:chat"}})

        # state: OutSchema = await self.app.ainvoke({"messages": msgs})

        messages = state["messages"]
        tool_outputs = state.get("tool_outputs", [])

        # Extract final assistant message (last AIMessage without tool_calls)
        final_text = ""
        for m in reversed(messages):
            if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
                final_text = m.content
                break

        # Optional: cache short transcript
        if self.redis and conversation_history is not None:
            try:
                key = f"agent:conversation:{user_id}"
                await self.redis.redis_client.lpush(key, json.dumps({
                    "user": user_message, "agent": final_text,
                    "timestamp": datetime.now().isoformat()
                }))
                await self.redis.redis_client.ltrim(key, 0, 19)
                await self.redis.redis_client.expire(key, 3600)
            except Exception:
                pass

        return AgentResponse(message=final_text or "Done.", tool_outputs=tool_outputs)
