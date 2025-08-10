from pathlib import Path
from typing import List, Optional, Dict
from ..client.es_client2 import ElasticsearchClient
from ..models.document import SearchResult

class Service:
    """Service layer for  operations"""
    
    def __init__(self):
        """Initialize service with Elasticsearch client"""
        self.client = ElasticsearchClient()
        
    def initialize_index(self, recreate: bool = False) -> None:
        """Initialize Elasticsearch index"""
        if not self.client.check_connection():
            raise ConnectionError("Could not connect to Elasticsearch")
        self.client.create_index(recreate=recreate)
        
    def index_dataset(self, dataset_path: str) -> None:
        """Index  data from dataset"""
        if not self.client.check_connection():
            raise ConnectionError("Could not connect to Elasticsearch")
            
        path = Path(dataset_path)
        if not path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
            
        self.client.index_dataset(path)
        
    def get_text_by_frame(self, video_id: str, frame_id: str)-> Optional[Dict[str, str]]:
        if not self.client.check_connection():
            raise ConnectionError("Could not connect to Elasticsearch")
        return self.client.get_text_by_frame(video_id=video_id, frame_id=frame_id)
        
    def search(
        self,
        query: str,
        size: int = 10,
        fields: List[str] = None,
        search_type: str = "combined"
    ) -> List[tuple]:
        """
        Search for text
        
        Args:
            query: Text to search for
            size: Number of results to return
            search_type: One of "exact", "fuzzy", or "combined"
            
        Returns:
            List of tuples (video_id, frame_id, score, filepath)
        """
        if not query.strip():
            return []
            
        if not self.client.check_connection():
            raise ConnectionError("Could not connect to Elasticsearch")
            
        results = self.client.search(query=query, size=size, fields=fields, search_type=search_type)
        # Extract raw scores
        raw_scores = [r.score for r in results]
        if not raw_scores:
            return []

        min_score = min(raw_scores)
        max_score = max(raw_scores)

        # Avoid division by zero
        if min_score == max_score:
            normalized = [1.0 for _ in raw_scores]
        else:
            normalized = [(s - min_score) / (max_score - min_score) for s in raw_scores]

        # Return normalized results
        return [
            (r.video_id, r.frame_id, norm_score, r.filepath)
            for r, norm_score in zip(results, normalized)
        ]