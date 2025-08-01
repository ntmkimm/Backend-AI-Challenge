import redis
import json
from typing import List
from models.schemas import Query
from services.clip_service import CLIPService
from services.milvus_service import MilvusService
from services.polar_service import PolarService
from core.utils import get_valid_queries, search_one_query
from concurrent.futures import ThreadPoolExecutor, as_completed
import zlib
import time

class RedisService:
    def __init__(self):
        print("Init Redis Service...")
        self._load_client()
        
    def _load_client(self):
        try:
            self.redis_client = redis.Redis(host='redis-server', port=6379, db=0, decode_responses=False)  # decode=False for binary (compressed)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {str(e)}")
    
    def make_cache_key(self, queries: List[Query]) -> str:
        """
        Generate a stable and JSON-safe cache key.
        """
        return json.dumps([json.loads(q.json()) for q in queries], sort_keys=True)

    def serialize_result(self, obj: any) -> bytes:
        """
        Compress and serialize result to store in Redis.
        """
        return zlib.compress(json.dumps(obj).encode("utf-8"))

    def deserialize_result(self, data: bytes) -> any:
        """
        Decompress and load Redis result.
        """
        return json.loads(zlib.decompress(data).decode("utf-8"))
    
    def get_all_answers_cached_redis(
        self,
        queries: List[Query],
        clip_service: CLIPService,
        milvus_service: MilvusService,
        polar_service: PolarService,
        ttl_seconds: int = 3600,
        max_workers: int = 8  # Tùy bạn chọn số threads
    ):
        text_queries = [q.text for q in queries]
        cache_key = self.make_cache_key(queries)

        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return self.deserialize_result(cached)
        except redis.RedisError as e:
            print(f"[Redis Warning] Failed to fetch from cache: {e}")

        # Cache miss -> proceed with real search
        start_time = time.time()
        embeddings = clip_service.encode_text_batch(text_queries)
        if not embeddings: embeddings = [None] * len(queries)
        end_time = time.time()
        print("Embedding time: ", end_time - start_time)
        
        start_time = end_time
        # Hàm phụ cho thread
        def _search(idx_q):
            stage_idx, q = idx_q
            return search_one_query(
                clip_service=clip_service,
                milvus_service=milvus_service,
                polar_service=polar_service,
                q=q,
                clip_embedding=embeddings[stage_idx]
            )
        # Dùng thread pool để chạy song song
        results = [None] * len(queries)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_search, (i, q)): i for i, q in enumerate(queries)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    print(f"[Search Error] Query #{idx} generated an exception: {exc}")
                    results[idx] = None  # hoặc xử lý fallback
        end_time = time.time()
        print("Search time", end_time - start_time)
        
        # Store into Redis
        try:
            self.redis_client.setex(cache_key, ttl_seconds, self.serialize_result(results))
        except redis.RedisError as e:
            print(f"[Redis Warning] Failed to save to cache: {e}")

        return results

