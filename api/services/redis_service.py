import redis
import json
from typing import List
from models.schemas import Query
from services.clip_service import CLIPService
from services.milvus_service import MilvusService
from services.polar_service import PolarService
from services.paraphrase_service import ParaphraseService
import zlib
import time
import asyncio
import hashlib
from core.module import search_one_query
from config.settings import REDIS_HOST, REDIS_PORT
import pickle

class RedisService:
    def __init__(self):
        print("Init Redis Service...")
        self._load_client()
        
    def _load_client(self):
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=False
            )  # decode=False for binary (compressed)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Redis: {str(e)}")
    
    def make_tmp_query_text_key_for_paraphrase(self, query_text: str):
        # Serialize và hash query
        """ QUERY PROCESS MODULE: sử dụng để lưu cache cho query.text cần paraphrase"""
        query_hash = hashlib.sha1(query_text.encode("utf-8")).hexdigest()
        return f"query_text_cache:{query_hash}"
    
    async def get_query_text_paraphrase_cached_redis(
        self,
        query_text: str,
        paraphrase_service: ParaphraseService,
        ttl_seconds: int = 90
    ):
        """
        QUERY PROCESS MODULE: sử dụng để lưu cache cho query.text cần paraphrase
        - Check cache theo key hash từ query_text.
        - Nếu có -> trả về JSON giải mã (list[str]).
        - Nếu miss -> gọi model paraphrase (không block event loop bằng to_thread/async method),
                      lưu lại Redis (JSON bytes), trả về kết quả.
        """
        key = self.make_tmp_query_text_key_for_paraphrase(query_text=query_text)

        # GET cache
        cached = await asyncio.to_thread(self.redis_client.get, key)
        if cached:
            try:
                return json.loads(cached.decode("utf-8"))  # list[str]
            except Exception as e:
                # Cache hỏng -> xóa để tính lại
                await asyncio.to_thread(self.redis_client.delete, key)

        # MISS cache -> chạy model
        try:
            # Nếu bạn đã có paraphrase_batch_async thì có thể:
            # result = (await paraphrase_service.paraphrase_batch_async([query_text]))[0]
            # Ở đây dùng paraphrase() và đẩy vào thread để không block:
            result = await asyncio.to_thread(paraphrase_service.paraphrase, query_text)  # list[str]
        except Exception as e:
            # Không lưu cache khi lỗi
            raise e

        # Lưu lại (JSON bytes)
        try:
            payload = json.dumps(result).encode("utf-8")
            await asyncio.to_thread(self.redis_client.setex, key, ttl_seconds, payload)
        except Exception as e:
            # Không fail toàn hàm nếu Redis set lỗi
            pass

        return result

    def make_tmp_search_result_key(self, user_id: str, queries: List[Query], mode: str = "normal"):
        # Serialize và hash query
        """ SEARCH ROUTER PANIGTION: sử dụng để lưu cache cho trạng thái queries Ở MODE SEARCH NÀO cho một ANSWERS đầy đủ TOP_K: panigtion"""
        queries_serialized = json.dumps([json.loads(q.json()) for q in queries], sort_keys=True)
        query_hash = hashlib.sha1(queries_serialized.encode("utf-8")).hexdigest()
        return f"search_cache:{user_id}:{mode}:{query_hash}"
    
    async def save_tmp_search_results_to_cache(self, redis_key, results, ttl_seconds=300):
        """ SEARCH ROUTER PANIGTION: sử dụng để lưu cache cho trạng thái queries results Ở MODE SEARCH NÀO cho một ANSWERS đầy đủ TOP_K: panigtion"""
        await asyncio.to_thread(
            self.redis_client.setex,
            redis_key,
            ttl_seconds,
            pickle.dumps(results)
        )
    
    def make_query_cache_key(self, query: Query) -> str:
        """
        SEARCH MODULE, SỬ DỤNG CHO HÀM GET_ANSWER*: 
        Chuẩn hóa cache key cho từng query (sort keys, exclude origin, sort obj nếu là list).
        """
        data = query.dict()
        s = json.dumps(data, sort_keys=True, separators=(',', ':'))
        # Hash để key ngắn gọn
        return "q:" + hashlib.sha1(s.encode('utf-8')).hexdigest()
    
    def serialize_result(self, obj: any) -> bytes:
        """ SEARCH MODULE, SỬ DỤNG CHO HÀM GET_ANSWER*: sử dụng cho cache KIỂU COMPRESS các kết quả của search """
        return zlib.compress(json.dumps(obj).encode("utf-8"))

    def deserialize_result(self, data: bytes) -> any:
        """ SEARCH MODULE, SỬ DỤNG CHO HÀM GET_ANSWER*: sử dụng cho decode cache KIỂU COMPRESS các kết quả của search """
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
        """SEARCH ROUTER: SỬ DỤNG EMBEDDING VÀ SEARCH SERVICE ĐỂ RETURN ANSWERS TRƯỚC KHI RERANKING"""
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

            # Tạo task cho từng query miss cache
            tasks = [
                search_one_query(
                    clip_service=clip_service,
                    milvus_service=milvus_service,
                    polar_service=polar_service,
                    q=queries[i]
                ) for j, i in enumerate(uncached_indices)
            ]
            
            fresh_results = []
            try:
                fresh_results = await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                print(f"[Search Error] Exception during search: {e}")
                fresh_results = [None] * len(uncached_indices)

            # Sắp xếp lại kết quả theo thứ tự ban đầu của queries
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
        SEARCH ROUTER: SỬ DỤNG EMBEDDING VÀ SEARCH SERVICE ĐỂ RETURN ANSWER CỦA MỘT STAGE
        Trả về kết quả search cho một query, sử dụng Redis cache. Nếu chưa có trong cache thì search, rồi lưu lại vào Redis.
        """
        key = self.make_query_cache_key(query)
        cached_bytes = await asyncio.to_thread(self.redis_client.get, key)
        if cached_bytes is not None:
            try:
                return self.deserialize_result(cached_bytes)
            except Exception as e:
                print(f"[Redis Warning] Deserialize failed: {e}")

        try:
            result = await search_one_query(
                clip_service=clip_service,
                milvus_service=milvus_service,
                polar_service=polar_service,
                q=query,
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