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
    
class InformationOfFrame(BaseModel):
    ocr_text: str = ''
    asr_text: str = ''
    objects: str = '' 

class ResultItem(BaseModel):
    id: str
    videoId: str
    title: str
    thumbnail: str
    confidence: float
    timestamp: str

