from typing import Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import json

# Root path for dataset
ROOT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1")

@dataclass
class Document:
    """Represents an document in Elasticsearch"""
    video_id: str
    frame_id: str
    ocr_text: str
    asr_text: str
    
    # @classmethod
    # def from_file(cls, video_dir: Path, frame_id: str, text: str) -> 'Document':
    #     """Create OCRDocument from file data"""
    #     return cls(
    #         video_id=video_dir.name,
    #         frame_id=frame_id,
    #         ocr_text=text.strip(),
    #         asr_text=asr.strip()
    #     )
    
    # def to_index_doc(self, index_name: str) -> Dict:
    #     """Convert to Elasticsearch index document"""
    #     return {
    #         '_op_type': 'index',
    #         "_index": index_name,
    #         "_id": f"{self.video_id}_{self.frame_id}",
    #         "_source": {
    #             "video_id": self.video_id,
    #             "frame_id": self.frame_id,
    #             "ocr_text": self.ocr_text,
    #             "asr_text": self.asr_text
    #         }
    #     }
    @classmethod
    def from_file(cls, video_dir, frame_id, ocr_text=None, asr_text=None):
        # video_id logic as before
        return cls(video_dir.name, frame_id, ocr_text, asr_text)

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
    filepath: str
    
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
        
        # Construct filepath using ROOT path
        filepath = str(ROOT / video_id / ("keyframes/keyframe_" + frame_id + ".webp"))
        
        return cls(
            video_id=video_id,
            frame_id=frame_id,
            score=hit['_score'],
            filepath=filepath
        ) 