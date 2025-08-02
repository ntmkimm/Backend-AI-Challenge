import redis
import json
from typing import List
from models.schemas import Query
from services.clip_service import CLIPService
from services.milvus_service import MilvusService
from services.polar_service import PolarService
from core.utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import zlib
import time
import asyncio
import hashlib
from config.settings import TOP_K
from utils.es_module import async_search_by_asr, async_search_by_ocr
from collections import defaultdict


async def search_one_query(
    clip_service: CLIPService,
    milvus_service: MilvusService,
    polar_service: PolarService,
    q: Query, 
    clip_embedding: List[float], 
    ):
    buffer = { 'text': None, 'ocr': None, 'asr': None, 'obj': None, 'origin': None, 'image': None}
    weighted_score = { 'text': 0.5, 'ocr': 0.5, 'asr': 0.2, 'obj': 0.1, 'origin': 0, 'image': 0.5 }
    
    # Chuẩn bị các task async / blocking
    tasks = []
    
    # Milvus text embedding search (async)
    if q.text and clip_embedding:
        tasks.append(milvus_service.search_by_embedding(clip_embedding))
    else:
        tasks.append(asyncio.sleep(0, result=None))

    # OCR search (async)
    if q.ocr:
        tasks.append(async_search_by_ocr(q.ocr, TOP_K))
    else:
        tasks.append(asyncio.sleep(0, result=None))

    # ASR search (async)
    if q.asr:
        tasks.append(async_search_by_asr(q.asr, TOP_K))
    else:
        tasks.append(asyncio.sleep(0, result=None))

    # Polar object search (blocking, chạy trên thread pool)
    if q.obj and q.obj[0]:
        tasks.append(polar_service.search_object(q.obj))
    else:
        tasks.append(asyncio.sleep(0, result=None))

    # Image embedding search (async)
    if q.image:
        try:
            img = base64_to_pil_image(base64_str=q.image)
        except Exception as e:
            print("Image decode failed:", e)
            img = None

        if img is not None:
            embedding = clip_service.encode_image(img)  # sync
            if embedding and len(embedding) > 0:
                tasks.append(milvus_service.search_by_embedding(embedding))
            else:
                tasks.append(asyncio.sleep(0, result=None))
        else:
            tasks.append(asyncio.sleep(0, result=None))
    else:
        tasks.append(asyncio.sleep(0, result=None))

    # Chạy song song tất cả các task
    results = await asyncio.gather(*tasks)

    buffer['text'], buffer['ocr'], buffer['asr'], buffer['obj'], buffer['image'] = results

    combined_results = defaultdict(lambda: {'score': 0.0, 'video_id': None, 'frame_id': None, 'filepath': None})

    # Xử lý kết quả từ buffer
    if buffer['text']:
        for h in buffer['text']:
            video_id = h.entity["video_id"]
            frame_id = h.entity["frame_id"]
            score = h.distance
            path = h.entity["filepath"]
            key = f"{video_id}_{frame_id}"
            combined_results[key].update({'video_id': video_id, 'frame_id': frame_id, 'filepath': path})
            combined_results[key]['score'] += weighted_score['text'] * score

    if buffer['image']:
        for h in buffer['image']:
            video_id = h.entity["video_id"]
            frame_id = h.entity["frame_id"]
            score = h.distance
            path = h.entity["filepath"]
            key = f"{video_id}_{frame_id}"
            combined_results[key].update({'video_id': video_id, 'frame_id': frame_id, 'filepath': path})
            combined_results[key]['score'] += weighted_score['image'] * score

    if buffer['ocr']:
        for video_id, frame_id, score, path in buffer['ocr']:
            key = f"{video_id}_{frame_id}"
            combined_results[key].update({'video_id': video_id, 'frame_id': int(frame_id), 'filepath': path})
            combined_results[key]['score'] += weighted_score['ocr'] * score

    if buffer['asr']:
        for video_id, frame_id, score, path in buffer['asr']:
            key = f"{video_id}_{frame_id}"
            combined_results[key].update({'video_id': video_id, 'frame_id': int(frame_id), 'filepath': path})
            combined_results[key]['score'] += weighted_score['asr'] * score

    obj_keys = {
        f"{row['video_id']}_{row['frame_id']}"
        for row in buffer['obj'].iter_rows(named=True)
    } if buffer['obj'] is not None and not buffer['obj'].is_empty() else set()

    if obj_keys:
        combined_results = {
            k: v for k, v in combined_results.items() if k in obj_keys
        }

    if not combined_results and obj_keys:
        for row in buffer['obj'].iter_rows(named=True):
            key = f"{row['video_id']}_{row['frame_id']}"
            combined_results[key] = {
                'video_id': row['video_id'],
                'frame_id': row['frame_id'],
                'filepath': row['filepath'],
                'score': weighted_score['obj'] * 1.0
            }

    return sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)[:TOP_K]

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