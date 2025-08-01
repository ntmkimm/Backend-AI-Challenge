from pydantic import BaseModel
from typing import List, Optional

class Query(BaseModel):
    text: str        
    asr: str         
    ocr: str         
    origin: str  
    obj: List[str]
    lang: str
    image: Optional[str] = ""

class ResultItem(BaseModel):
    id: str
    videoId: str
    title: str
    thumbnail: str
    confidence: float
    timestamp: str

class OCRSearchRequest(BaseModel):
    query: str
    size: int = 10

class NearbyFramesRequest(BaseModel):
    video_id: str
    frame_id: int
    window_size: int = 5  # Number of frames to fetch on each side 

class ImageSearchRequest(BaseModel):
    image_base64: str  # Base64 encoded image
    k: int = 10  # Number of results to return

class ImageSearchResult(BaseModel):
    score: float
    image_path: str
    index_id: int 
