from pymilvus import Collection, connections
from typing import List, Dict, Any
from config.settings import MILVUS_HOST, MILVUS_PORT, TOP_K
import asyncio

class MilvusService:
    def __init__(self, collection_name):
        print(f"Init Milvus {collection_name} Service...")
        self._connect()
        self.collection = self._load_collection(collection_name=collection_name)

    def _connect(self):
        try:
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")

    def _load_collection(self, collection_name) -> Collection:
        try:
            collection = Collection(collection_name)
            collection.load()
            return collection
        except Exception as e:
            raise ConnectionError(f"Failed to load collection: {str(e)}")

    def get_search_iterator(self, embedding, batch_size = 200):
        return self.collection.search_iterator(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=TOP_K,  
            batch_size=batch_size,
            output_fields=["frame_id", "video_id"],
        )
    
    async def search_by_embedding(self, embedding: List[float], limit: int = TOP_K) -> List[Dict[str, Any]]:
        def blocking_search():
            results = self.collection.search(
                data=[embedding],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=limit,
                output_fields=["frame_id", "video_id"],
            )
            # Flatten results for return, each result is a list (one per query vector)
            # Since data=[embedding], we have only one result list
            hits = results[0]
            return [hit.to_dict() for hit in hits]
        results = await asyncio.to_thread(blocking_search)
        return results