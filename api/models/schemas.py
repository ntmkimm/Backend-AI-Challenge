from pydantic import BaseModel
from typing import List, Optional, Dict
from pydantic import Field
class Query(BaseModel):
    text: str        
    asr: str         
    ocr: str         
    origin: str  
    obj: List[str]
    lang: str
    image: Optional[str] = ""
    
class InformationOfFrame(BaseModel):
    ocr_text: str = ''
    asr_text: str = ''
    objects: str = '' 

class ResultItem(BaseModel):
    id: str
    videoId: str
    confidence: float
    timestamp: str
    
class HistoryItem(BaseModel):
    queries: List[Query]
    
class ModelProvider(BaseModel):
    clip: Optional[bool] = True
    beit3: Optional[bool] = True 
    siglip2: Optional[bool] = True
    

class ReverseObjectFilterResponse(BaseModel):
    # Maps object name (str) to a list of frame keys (str) where it appears
    objects_to_frame_keys: Dict[str, List[str]]
    # Still useful to have global counts for sorting/display
    global_object_counts: Dict[str, int]
    
class DDGQuery(BaseModel):
    query: str = Field(..., description="Search query string")
    max_results: int = Field(5, ge=1, le=25, description="Number of results to return (1-25)")
    region: str = Field("wt-wt", description="Region code, e.g., 'us-en', 'wt-wt' (worldwide)")
    time: str = Field("", description="Time limit: '', 'd', 'w', 'm', 'y'")
    safesearch: str = Field("moderate", description="'off' | 'moderate' | 'strict'")

class DDGResult(BaseModel):
    title: str
    link: str
    snippet: str
class DDGImageQuery(BaseModel):
    """Request model for DuckDuckGo Image Search."""
    query: str
    region: str = Field(default="wt-wt", description="Worldwide")
    safesearch: str = Field(default="moderate", description="'off', 'moderate', or 'strict'")
    max_results: int = Field(default=10, description="Max number of images to return")

class DDGImageResult(BaseModel):
    """Response model for a single DuckDuckGo Image result."""
    title: str
    image: str  # URL to the full-size image
    thumbnail: str # URL to the thumbnail
    url: str # URL of the source page
    height: int
    width: int
    source: str
