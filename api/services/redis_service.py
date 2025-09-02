import redis.asyncio as redis
import json
import zlib
import hashlib
import pickle
from typing import List
from models.schemas import Query
from core.module import search_one_query
from config.settings import REDIS_HOST, REDIS_PORT
import asyncio

class RedisService:
    def __init__(self):
        print("Init Redis Service")
        self.redis_client = None

    async def async_init(self):
        self.redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=0,
            decode_responses=False
        )
        return self
    
    async def flush_user_search_cache(self, user_id: str, mode: str):
        """
        Xoá toàn bộ cache có prefix 'search_cache:{user_id}:{mode}:*'
        """
        pattern = f"search_cache:{user_id}:{mode}:*"
        cursor = b"0"
        keys_deleted = 0

        while cursor:
            cursor, keys = await self.redis_client.scan(cursor=cursor, match=pattern, count=1000)
            if keys:
                await self.redis_client.delete(*keys)
                keys_deleted += len(keys)

    async def get_dislike_labels(self, user_id: str = 'anynomous') -> list[int]:
        default_cluster = [2, 17, 39, 56, 57, 60, 63, 65, 78, 79, 90, 92, 93, 95, 98, 111, 113, 121, 124, 132, 146, 147, 150, 161, 166, 169, 176, 192]
        labels = await self.redis_client.smembers(f"cluster:{user_id}")
        return [int(l) for l in labels] + default_cluster

    def make_tmp_search_result_key(self, user_id: str, queries: List[Query], mode: str = "normal") -> str:
        queries_serialized = json.dumps([json.loads(q.json()) for q in queries], sort_keys=True)
        query_hash = hashlib.sha1(queries_serialized.encode("utf-8")).hexdigest()
        return f"search_cache:{user_id}:{mode}:{query_hash}"

    async def save_tmp_search_results_to_cache(self, redis_key, results, ttl_seconds=300):
        await self.redis_client.setex(redis_key, ttl_seconds, pickle.dumps(results))
        
    async def add_queries_to_history(self, queries: Query, dislike_labels: List, user_id: str = 'anynomous'):
        key = f"history:{user_id}"
        await self.redis_client.lpush(key, json.dumps({
            "queries": [q.dict() for q in queries],
            "dislikes": dislike_labels
        }))
        await self.redis_client.ltrim(key, 0, 10)
        
    async def get_queries_history(self, user_id: str = 'anynomous', limit = 10):
        key = f"history:{user_id}"
        items = await self.redis_client.lrange(key, 0, limit - 1) 

        history = []
        for raw in items:
            try:
                history.append(json.loads(raw))
            except Exception as e:
                print(f"[Redis Warning] Failed to parse history entry: {e}")
        return history

    def make_query_cache_key(self, query: Query, user_id: str = 'anynomous') -> str:
        data = query.dict()
        s = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return f"q:{user_id}:" + hashlib.sha1(s.encode('utf-8')).hexdigest()

    def serialize_result(self, obj: any) -> bytes:
        return zlib.compress(json.dumps(obj).encode("utf-8"))

    def deserialize_result(self, data: bytes) -> any:
        return json.loads(zlib.decompress(data).decode("utf-8"))

    async def get_all_answers_cached_redis(
        self,
        queries: List[Query],
        user_id: str = 'anynomous',
        ttl_seconds: int = 90,
    ):
        keys = [self.make_query_cache_key(query=q, user_id=user_id) for q in queries]
        cached_bytes_list = await self.redis_client.mget(keys)

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

        if uncached_indices:
            print(f"Cache miss at: {uncached_indices}")
            dislike_labels = await self.get_dislike_labels(user_id=user_id)

            # add to history
            await self.add_queries_to_history(queries=queries, dislike_labels=dislike_labels, user_id=user_id)
            history = await self.get_queries_history(user_id=user_id)
            print(history)
            tasks = [
                search_one_query(
                    q=queries[i],
                    dislike_labels=dislike_labels
                ) for i in uncached_indices
            ]

            try:
                fresh_results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                print(f"[Search Error] Exception during search: {e}")
                fresh_results = [None] * len(uncached_indices)

            for j, idx in enumerate(uncached_indices):
                res = fresh_results[j]
                if isinstance(res, Exception):
                    print(f"[Search Error] Query #{idx} generated an exception: {res}")
                    results[idx] = None
                    continue
                results[idx] = res
                try:
                    await self.redis_client.setex(
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
        user_id: str = 'anynomous',
        ttl_seconds: int = 90,
    ):
        key = self.make_query_cache_key(query=query, user_id=user_id)
        cached_bytes = await self.redis_client.get(key)
        if cached_bytes is not None:
            try:
                return self.deserialize_result(cached_bytes)
            except Exception as e:
                print(f"[Redis Warning] Deserialize failed: {e}")

        try:
            dislike_labels = await self.get_dislike_labels(user_id=user_id)
            result = await search_one_query(
                q=query,
                dislike_labels=dislike_labels
            )
        except Exception as e:
            print(f"[Search Error] Exception during search_one_query: {e}")
            result = None

        if result is not None:
            try:
                await self.redis_client.setex(
                    key,
                    ttl_seconds,
                    self.serialize_result(result)
                )
            except Exception as e:
                print(f"[Redis Warning] Failed to save to cache for key {key}: {e}")

        return result
