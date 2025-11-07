from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field
from .tools import AgentSearchQuery
class LCQuery(BaseModel):
    type: Literal["text", "image"] = Field(..., description="Query type")
    value: str = Field(..., description="Search value")

class LCAgentJSON(BaseModel):
    message: str = Field(..., description="Assistant’s friendly reply to the user.")
    queries: List[AgentSearchQuery] = Field(default_factory=list,
        description="List of search operations. Each item represents a step in time. A single item can have multiple fields for a composite search."
    )
    should_search: bool = Field(..., description="Whether to trigger a search.")
    metadata: Optional[Dict] = None
