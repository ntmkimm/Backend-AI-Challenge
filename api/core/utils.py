from fastapi import APIRouter, HTTPException, Depends
from typing import List
from collections import defaultdict
import torch
import heapq

from models.schemas import Query
from services.clip_service import CLIPService
from services.milvus_service import MilvusService
from services.polar_service import PolarService
from config.settings import MAX_FRAME_GAP, MEDIA_SERVER_URL, TOP_K
from utils.es_module import async_search_by_asr, async_search_by_ocr
from collections import defaultdict
from typing import List
from models.schemas import Query
from PIL import Image
from io import BytesIO
import base64
import cv2
import numpy as np
import json 
import asyncio

from typing import List
from models.schemas import Query

def get_valid_queries(queries: List[Query]) -> List[Query]:
    res = []
    for q in queries:
        if not q.text and not q.ocr and not q.asr and not q.origin and not q.obj and not q.image:
            continue
        res.append(q)
    return res

def base64_to_cv2_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def base64_to_pil_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    img = img.convert("RGB")  # Bắt buộc chuyển về RGB cho model encode (an toàn)
    return img

async def search_one_query(
    clip_service: CLIPService,
    milvus_service: MilvusService,
    polar_service: PolarService,
    q: Query, 
    clip_embedding: List[float], 
    ):
    """
    class Query(BaseModel):
        text: str        
        asr: str         
        ocr: str         
        origin: str  
        obj: List[str]
        lang: str
        image: str
    
    clip_embeddings: đây là embedding được encode của Query.text đó
        
    Đây là hàm xử lí 1 Query, return về 1 hits tổng hợp theo weight score (hits này sẽ được append vào all_answers trong search module)
    
    Note các hits của từng service:
    Kết quả TOP_K của search ocr: for (video_id, frame_id (keyframe_X.webp), score, path) in hits
    Kết quả TOP_K của search milvus: h.entity["video_id"], h.entity["frame_id"], h.distance, h.entity["filepath"] for h in hits
    Kết quả của polar filter: hits[[video_id frame_id filepath]]
    """
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