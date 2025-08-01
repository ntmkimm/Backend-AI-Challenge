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
from utils.es_module import search_by_ocr, search_by_asr
from collections import defaultdict
from typing import List
from models.schemas import Query
from PIL import Image
from io import BytesIO
import base64
import cv2
import numpy as np
import json 

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

def search_one_query(
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
    r = 0
    r += 1
    print(r)
    if q.text and clip_embedding:
        buffer['text'] = milvus_service.search_by_embedding(clip_embedding) 
    r += 1
    print(r)
    if q.ocr:
        buffer['ocr'] = search_by_ocr(q.ocr, TOP_K) 
    r += 1
    print(r)
    if q.asr:
        buffer['asr'] = search_by_asr(q.asr, TOP_K) 
    r += 1
    print(r)
    if q.obj and q.obj[0]:
        buffer['obj'] = polar_service.search_object(q.obj) 
    r += 1
    print(r)
    if q.origin:
        pass
    r += 1
    print(r)
    if q.image:
        img = base64_to_pil_image(base64_str=q.image)
        if img is not None:
            embedding = clip_service.encode_image(img)
            if embedding is not None and len(embedding) > 0:
                buffer['image'] = milvus_service.search_by_embedding(embedding)
        else:
            print("Image decode failed: Không phải ảnh hoặc base64 lỗi")
    r += 1
    print(r)
    combined_results = defaultdict(lambda: {'score': 0.0, 'video_id': None, 'frame_id': None, 'filepath': None})

    # Text results (Milvus)
    if buffer['text']:
        for h in buffer['text']:
            video_id = h.entity["video_id"]
            frame_id = h.entity["frame_id"]
            score = h.distance  
            path = h.entity["filepath"]
            key = f"{video_id}_{frame_id}"
            combined_results[key].update({'video_id': video_id, 'frame_id': frame_id, 'filepath': path})
            combined_results[key]['score'] += weighted_score['text'] * score
    r += 1
    print(r)
    # Image results (Milvus)
    if buffer['image']:
        for h in buffer['image']:
            video_id = h.entity["video_id"]
            frame_id = h.entity["frame_id"]
            score = h.distance
            path = h.entity["filepath"]
            key = f"{video_id}_{frame_id}"
            combined_results[key].update({'video_id': video_id, 'frame_id': frame_id, 'filepath': path})
            combined_results[key]['score'] += weighted_score['image'] * score
    r += 1
    print(r)
    # OCR results (Elastic Search)
    if buffer['ocr']:
        for video_id, frame_id, score, path in buffer['ocr']:
            frame_id_int = int(frame_id[9:-5])
            key = f"{video_id}_{frame_id_int}"
            combined_results[key].update({'video_id': video_id, 'frame_id': frame_id_int, 'filepath': path})
            combined_results[key]['score'] += weighted_score['ocr'] * score
    r += 1
    print(r)
    # ASR results (Elastic Search)
    print(buffer['asr'])
    # if buffer['asr']:
    #     for video_id, frame_id, score, path in buffer['asr']:
    #         # frame_id của asr là "xxx" với xxx là number
    #         # frame_id của ocr là "keyframe_xxx.webp"
    #         frame_id_int = int(frame_id)
    #         key = f"{video_id}_{frame_id_int}"
    #         combined_results[key].update({'video_id': video_id, 'frame_id': frame_id_int, 'filepath': path})
    #         combined_results[key]['score'] += weighted_score['asr'] * score
    r += 1
    print(r)
    # Object results (Polar filtering)
    obj_keys = {
        f"{row['video_id']}_{row['frame_id']}"
        for row in buffer['obj'].iter_rows(named=True)
    } if buffer['obj'] is not None and not buffer['obj'].is_empty() else set()
    r += 1
    print(r)
    if obj_keys:
        combined_results = {
            k: v for k, v in combined_results.items() if k in obj_keys
        }
    r += 1
    print(r)
    # Fallback: nếu không có kết quả nào, nhưng obj có thì trả về obj thôi
    if not combined_results and obj_keys:
        for row in buffer['obj'].iter_rows(named=True):
            key = f"{row['video_id']}_{row['frame_id']}"
            combined_results[key] = {
                'video_id': row['video_id'], 
                'frame_id': row['frame_id'],
                'filepath': row['filepath'],
                'score': weighted_score['obj'] * 1.0
            }
    r += 1
    print(r)
    return sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)[:TOP_K]