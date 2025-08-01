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
import asyncio

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
    
    async def get_all_answers_cached_redis(
        self,
        queries: List[Query],
        clip_service: CLIPService,
        milvus_service: MilvusService,
        polar_service: PolarService,
        ttl_seconds: int = 3600,
        max_workers: int = 8,
    ):
        text_queries = [q.text if q.text else "" for q in queries]
        cache_key = self.make_cache_key(queries)
        start_time = time.time()
        # try:
        #     cached = await asyncio.to_thread(self.redis_client.get, cache_key)
        #     if cached:
        #         return self.deserialize_result(cached)
        # except Exception as e:
        #     print(f"[Redis Warning] Failed to fetch from cache: {e}")
        end_time = time.time()
        print("Time Redis Search: ", end_time - start_time)
        if text_queries and any(x.strip() for x in text_queries):
            t0 = time.time()
            embeddings = clip_service.encode_text_batch(text_queries)
            print("Time for embedding:", time.time() - t0)
        else:
            t0 = time.time()
            embeddings = [None] * len(queries)
            print("Time for [None]*len(queries):", time.time() - t0)

        start_time = time.time()

        # Tạo các coroutine async, gọi trực tiếp search_one_query đã async
        tasks = [
            search_one_query(
                clip_service=clip_service,
                milvus_service=milvus_service,
                polar_service=polar_service,
                q=q,
                clip_embedding=embeddings[i]
            )
            for i, q in enumerate(queries)
        ]

        results = []
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            print(f"[Search Error] Exception during gathering search tasks: {e}")

        # Xử lý lỗi từng task nếu có Exception
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                print(f"[Search Error] Query #{i} generated an exception: {res}")
                results[i] = None

        end_time = time.time()
        print("Search time", end_time - start_time)

        try:
            await asyncio.to_thread(
                self.redis_client.setex, cache_key, ttl_seconds, self.serialize_result(results)
            )
        except Exception as e:
            print(f"[Redis Warning] Failed to save to cache: {e}")

        return results

