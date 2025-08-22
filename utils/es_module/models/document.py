from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class Document:
    """Represents an document in Elasticsearch"""
    video_id: str
    frame_id: str
    ocr_text: str
    asr_text: str
    
    @classmethod
    def from_file(cls, video_dir: str, frame_id: str, ocr_text: str=None, asr_text: str=None):
        # video_id logic as before
        return cls(video_dir, frame_id, ocr_text, asr_text)

    def to_index_doc(self, index_name):
        doc = {
            '_op_type': 'index',
            '_index': index_name,
            'video_id': self.video_id,
            'frame_id': self.frame_id,
        }
        if self.ocr_text is not None:
            doc['ocr_text'] = self.ocr_text
        if self.asr_text is not None:
            doc['asr_text'] = self.asr_text
        return doc


@dataclass
class SearchResult:
    """Represents a search result from Elasticsearch"""
    video_id: str
    frame_id: str
    score: float
    
    @property
    def unique_id(self) -> str:
        """Get unique identifier for deduplication"""
        return f"{self.video_id}_{self.frame_id}"
    
    @property
    def result_path(self) -> str:
        """Get result path in format video_id/frame_id"""
        return f"{self.video_id}/{self.frame_id}"
    
    @classmethod
    def from_es_hit(cls, hit: Dict) -> 'SearchResult':
        """Create SearchResult from Elasticsearch hit"""
        source = hit['_source']
        frame_id = source['frame_id']
        video_id = source['video_id']
        
        return cls(
            video_id=video_id,
            frame_id=frame_id,
            score=hit['_score'],
            # filepath=filepath
        ) 