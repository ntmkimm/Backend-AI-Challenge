from core.utils import *
from config.settings import TOP_K
from utils.es_module import async_search_by_asr, async_search_by_ocr
from collections import defaultdict
from services.clip_service import CLIPService
from services.milvus_service import MilvusService
from services.polar_service import PolarService
from models.schemas import Query

import asyncio
from typing import List

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
        print("search text")
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
