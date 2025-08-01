from pymilvus import Collection, connections
from typing import List, Dict, Any
from config.settings import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, TOP_K
import asyncio

class MilvusService:
    def __init__(self):
        print("Init Milvus Service...")
        self._connect()
        self.collection = self._load_collection()

    def _connect(self):
        try:
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")

    def _load_collection(self) -> Collection:
        try:
            collection = Collection(COLLECTION_NAME)
            collection.load()
            return collection
        except Exception as e:
            raise ConnectionError(f"Failed to load collection: {str(e)}")

    async def search_by_embedding(self, embedding: List[float], limit: int = TOP_K) -> List[Dict[str, Any]]:
        def blocking_search():
            iterator = self.collection.search_iterator(
                data=[embedding],
                anns_field="clip_embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=limit,
                batch_size=200,
                output_fields=["filepath", "frame_id", "video_id"],
            )
            results = []
            while True:
                hits = iterator.next()
                if not hits:
                    iterator.close()
                    break
                results.extend(hits)
            return results
        results = await asyncio.to_thread(blocking_search)
        return results

    def get_frames_by_video_id(self, video_id: str) -> List[Dict[str, Any]]:
        expr = f'video_id == "{video_id}"'
        results = self.collection.query(
            expr=expr,
            output_fields=["filepath", "frame_id", "video_id"],
            consistency_level="Strong"
        )
        return results 