import redis
import json
from typing import List
from models.schemas import Query
from services.clip_service import CLIPService
from services.milvus_service import MilvusService
from services.polar_service import PolarService
import zlib
import time
import asyncio
import hashlib
from core.module import search_one_query
import pickle

class RedisService:
    def __init__(self):
        print("Init Redis Service...")
        self._load_client()
        
    def _load_client(self):
        try:
            self.redis_client = redis.Redis(
                host='redis-server', port=6379, db=0, decode_responses=False
            )  # decode=False for binary (compressed)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {str(e)}")

    def make_query_cache_key(self, query: Query) -> str:
        """
        Chuẩn hóa cache key cho từng query (sort keys, exclude origin, sort obj nếu là list).
        """
        data = query.dict()
        data.pop("origin", None)
        s = json.dumps(data, sort_keys=True, separators=(',', ':'))
        # Hash để key ngắn gọn
        return "q:" + hashlib.sha1(s.encode('utf-8')).hexdigest()

    def serialize_result(self, obj: any) -> bytes:
        return zlib.compress(json.dumps(obj).encode("utf-8"))

    def deserialize_result(self, data: bytes) -> any:
        return json.loads(zlib.decompress(data).decode("utf-8"))
    
    def make_tmp_search_result_key(self, user_id: str, queries: List[Query], mode: str = "normal"):
        # Serialize và hash query
        queries_serialized = json.dumps(
            [q.dict(exclude={"origin"}) for q in queries],
            sort_keys=True
        )
        query_hash = hashlib.sha1(queries_serialized.encode("utf-8")).hexdigest()
        return f"search_cache:{user_id}:{mode}:{query_hash}"
    
    async def save_tmp_search_results_to_cache(self, redis_key, results, ttl_seconds=300):
        await asyncio.to_thread(
            self.redis_client.setex,
            redis_key,
            ttl_seconds,
            pickle.dumps(results)
        )
    

    async def get_all_answers_cached_redis(
        self,
        queries: List[Query],
        clip_service: CLIPService,
        milvus_service: MilvusService,
        polar_service: PolarService,
        ttl_seconds: int = 3600,
        max_workers: int = 8,
    ):
        keys = [self.make_query_cache_key(q) for q in queries]

        # Redis mget, trong asyncio nên bọc bằng asyncio.to_thread
        cached_bytes_list = await asyncio.to_thread(self.redis_client.mget, keys)

        # Giải nén các kết quả đã cache
        results = []
        uncached_indices = []
        for idx, cached_bytes in enumerate(cached_bytes_list):
            if cached_bytes is not None:
                try:
                    results.append(self.deserialize_result(cached_bytes))
                except Exception as e:
                    print(f"[Redis Warning] Deserialize failed at idx {idx}: {e}")
                    results.append(None)
                    uncached_indices.append(idx)
            else:
                results.append(None)
                uncached_indices.append(idx)

        # Chỉ search lại những query bị miss cache
        if uncached_indices:
            print(f"Cache miss at: {uncached_indices}")
            # Tạo text_queries (batch) cho những cái cần embed text
            text_queries = [
                queries[i].text if queries[i].text else ""
                for i in uncached_indices
            ]
            if text_queries and any(x.strip() for x in text_queries):
                t0 = time.time()
                embeddings = clip_service.encode_text_batch(text_queries)
                print("Time for embedding:", time.time() - t0)
            else:
                embeddings = [None] * len(uncached_indices)

            # Tạo task cho từng query miss cache
            tasks = [
                search_one_query(
                    clip_service=clip_service,
                    milvus_service=milvus_service,
                    polar_service=polar_service,
                    q=queries[i],
                    clip_embedding=embeddings[j]
                )
                for j, i in enumerate(uncached_indices)
            ]
            fresh_results = []
            try:
                fresh_results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                print(f"[Search Error] Exception during search: {e}")
                fresh_results = [None] * len(uncached_indices)

            # Lưu cache cho những query này
            for j, idx in enumerate(uncached_indices):
                res = fresh_results[j]
                # Nếu lỗi, không lưu cache
                if isinstance(res, Exception):
                    print(f"[Search Error] Query #{idx} generated an exception: {res}")
                    results[idx] = None
                    continue
                results[idx] = res
                try:
                    await asyncio.to_thread(
                        self.redis_client.setex,
                        keys[idx],
                        ttl_seconds,
                        self.serialize_result(res)
                    )
                except Exception as e:
                    print(f"[Redis Warning] Failed to save to cache for key {keys[idx]}: {e}")

        return results
    
    async def get_one_answer_cached_redis(
        self,
        query: Query,
        clip_service: CLIPService,
        milvus_service: MilvusService,
        polar_service: PolarService,
        ttl_seconds: int = 3600,
    ):
        """
        Trả về kết quả search cho một query, sử dụng Redis cache. Nếu chưa có trong cache thì search, rồi lưu lại vào Redis.
        """
        key = self.make_query_cache_key(query)
        cached_bytes = await asyncio.to_thread(self.redis_client.get, key)
        if cached_bytes is not None:
            try:
                return self.deserialize_result(cached_bytes)
            except Exception as e:
                print(f"[Redis Warning] Deserialize failed: {e}")

        # Nếu miss cache hoặc lỗi giải nén: thực hiện search lại
        # Chuẩn bị text embedding nếu có text
        clip_embedding = None
        if query.text and query.text.strip():
            t0 = time.time()
            clip_embedding = clip_service.encode_text(query.text)
            print("Time for embedding one:", time.time() - t0)

        try:
            result = await search_one_query(
                clip_service=clip_service,
                milvus_service=milvus_service,
                polar_service=polar_service,
                q=query,
                clip_embedding=clip_embedding
            )
        except Exception as e:
            print(f"[Search Error] Exception during search_one_query: {e}")
            result = None

        # Lưu cache nếu search thành công (không lỗi)
        if result is not None:
            try:
                await asyncio.to_thread(
                    self.redis_client.setex,
                    key,
                    ttl_seconds,
                    self.serialize_result(result)
                )
            except Exception as e:
                print(f"[Redis Warning] Failed to save to cache for key {key}: {e}")

        return result