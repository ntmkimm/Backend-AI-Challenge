# routers/agent.py
from typing import Any, List
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from dependencies.services import get_agent_controller, get_quick_agent_controller

router = APIRouter(prefix="/agent", tags=["agent"])

class ChatRequest(BaseModel):
    msg: str
    user_id: str = "anonymous"

class ChatResponse(BaseModel):
    message: str
    tool_outputs: List[Any] = []

@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, agent = Depends(get_agent_controller)):
    res = await agent.process_message(user_message=payload.msg, user_id=payload.user_id)
    return ChatResponse(message=res.message, tool_outputs=res.tool_outputs)
@router.post("/chat/quick", response_model=ChatResponse)
async def quick_chat(payload: ChatRequest, agent = Depends(get_quick_agent_controller)):
    res = await agent.process_message(user_message=payload.msg, user_id=payload.user_id)
    return ChatResponse(message=res.message, tool_outputs=res.tool_outputs)