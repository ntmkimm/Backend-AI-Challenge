from core.utils import *
from config.settings import TOP_K
from utils.es_module import async_search_by_asr, async_search_by_ocr
from collections import defaultdict
from dependencies.services import service_manager
from models.schemas import Query, ModelProvider
from services.milvus_service import MilvusService
from services.clip_service import CLIPService
from services.postgres_service import get_connection, release_connection
from config.settings import OPENCLIP_MILVUS, BEIT3_MILVUS, DEVICE_0, DEVICE_1, SIGLIP2_MILVUS

import asyncio
from typing import List, Dict, Any
import time

async def search_by_text(
    embedding_service,                
    milvus_service: MilvusService,             
    text: str,
):
    if not text or not embedding_service or not milvus_service:
        return None
    t0 = time.time()
    embedding = await asyncio.to_thread(embedding_service.encode_single_text, text)
    t1 = time.time()
    results = await milvus_service.search_by_embedding(embedding)
    t2 = time.time()
    print("time embed: ", t1 - t0)
    print("time search milvus: ", t2 - t1)

    return results

async def search_by_image(
    embedding_service,                # CLIPService
    milvus_service: MilvusService,              # MilvusService
    image: Image.Image,          # already decoded PIL Image (RGB preferred)
):

    if image is None or embedding_service is None or milvus_service is None:
        return None
    t0 = time.time()
    embedding = await asyncio.to_thread(embedding_service.encode_image, image)
    t1 = time.time()
    results = await milvus_service.search_by_embedding(embedding)
    t2 = time.time()
    print("time embed: ", t1 - t0)
    print("time search milvus: ", t2 - t1)

    return results

async def search_by_dislike_labels(
    labels: List[int],
) -> Dict:
    """
    Given results [{'video_id': str, 'frame_id': int, 'score': float}, ...]
    keep only those with cluster.label in labels.
    """
    if not labels:
        return {}

    # Extract (video_id, frame_id) pairs
    t0 = time.time()  
    conn = get_connection()
    cur = conn.cursor()
    try:
        # Query all matching rows
        cur.execute(
            """
            SELECT video_id, frame_id
            FROM cluster
            WHERE label = ANY(%s)
            """,
            (labels,)
        )
        rows = cur.fetchall()
        matched_pairs = {f"{vid}_{fid}" for vid, fid in rows}
    finally:
        cur.close()
        release_connection(conn)
    print("time for query postgres: ", time.time() - t0)
    # Keep only results present in allowed set
    return matched_pairs

async def search_one_query(
    q: Query,
    dislike_labels: List[int] = None,
    model_provider: ModelProvider = None,
):
    # service
    print("dislike labels: ", dislike_labels)
    polar_service = service_manager.get_polar_service()
    milvus_services = service_manager.get_milvus_services()
    
    siglip2_service, clip_service, beit3_service = None, None, None
    if model_provider.siglip2:
        # siglip2_service = service_manager.get_siglip2_service(device=DEVICE_0)
        siglip2_service = None
    if model_provider.clip:
        clip_service = service_manager.get_clip_service(device=DEVICE_0)
        # clip_service = None
    if model_provider.beit3:
        # beit3_service = service_manager.get_beit3_service(device=DEVICE_1)
        beit3_service = None

    weighted_score = { 'clip': 0, 'beit3': 0, 'siglip2': 0, 'ocr': 0, 'asr': 0, 'obj': 0, 'image': 0 }
    if clip_service: weighted_score['clip'] = 0.2
    if beit3_service: weighted_score['beit3'] = 0.2
    if siglip2_service: weighted_score['siglip2'] = 0.2
    full_score = weighted_score['clip'] + weighted_score['beit3'] + weighted_score['siglip2']
    weighted_score['ocr'] = 2 / 3 * full_score
    weighted_score['asr'] = 1 / 3 * full_score
    weighted_score['image'] = full_score
    
    # Chuẩn bị các task async / blocking
    start_time = time.time()
    tasks = []
    
    if q.text:
        tasks.append(search_by_text(siglip2_service, milvus_services[SIGLIP2_MILVUS], q.text))
        tasks.append(search_by_text(clip_service, milvus_services[OPENCLIP_MILVUS], q.text))
        tasks.append(search_by_text(beit3_service, milvus_services[BEIT3_MILVUS], q.text))
    else:
        tasks.append(asyncio.sleep(0, result=None))
        tasks.append(asyncio.sleep(0, result=None))
        tasks.append(asyncio.sleep(0, result=None))     
        
    # Image embedding search (async)
    if q.image:
        try:
            img = base64_to_pil_image(base64_str=q.image)
        except Exception as e:
            print("Image decode failed:", e)
            img = None

        if img is not None:
            tasks.append(search_by_image(clip_service, milvus_services[OPENCLIP_MILVUS], img))
            # if model_provider.clip:
            #     tasks.append(search_by_image(clip_service, milvus_services[OPENCLIP_MILVUS], img))
            # elif model_provider.beit3:
            #     tasks.append(search_by_image(beit3_service, milvus_services[BEIT3_MILVUS], img))
            # elif model_provider.siglip2:
            #     tasks.append(search_by_image(siglip2_service, milvus_services[SIGLIP2_MILVUS], img))
        else:
            tasks.append(asyncio.sleep(0, result=None))
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
    
    if dislike_labels:
        tasks.append(search_by_dislike_labels(labels=dislike_labels))
    else:
        tasks.append(asyncio.sleep(0, result=None))

    # Chạy song song tất cả các task
    results = await asyncio.gather(*tasks)
    
    # get buffer
    buffer = {}
    buffer['siglip2'], buffer['openclip'], buffer['beit3'], buffer['image'], buffer['ocr'], buffer['asr'], buffer['obj'], buffer['dislike_labels'] = results

    # Xử lý kết quả từ buffer
    combined_results = defaultdict(lambda: {'score': 0.0, 'video_id': None, 'frame_id': None })
  
    if buffer['siglip2']:
        for h in buffer['siglip2']:
            video_id = h.entity["video_id"]
            frame_id = h.entity["frame_id"]
            score = h.distance
            key = f"{video_id}_{frame_id}"
            combined_results[key].update({'video_id': video_id, 'frame_id': frame_id})
            combined_results[key]['score'] += weighted_score['siglip2'] * score
  
    if buffer['beit3']:
        for h in buffer['beit3']:
            video_id = h.entity["video_id"]
            frame_id = h.entity["frame_id"]
            score = h.distance
            key = f"{video_id}_{frame_id}"
            combined_results[key].update({'video_id': video_id, 'frame_id': frame_id})
            combined_results[key]['score'] += weighted_score['beit3'] * score

    if buffer['openclip']:
        for h in buffer['openclip']:
            video_id = h.entity["video_id"]
            frame_id = h.entity["frame_id"]
            score = h.distance
            key = f"{video_id}_{frame_id}"
            combined_results[key].update({'video_id': video_id, 'frame_id': frame_id})
            combined_results[key]['score'] += weighted_score['clip'] * score

    if buffer['image']:
        for h in buffer['image']:
            video_id = h.entity["video_id"]
            frame_id = h.entity["frame_id"]
            score = h.distance
            key = f"{video_id}_{frame_id}"
            combined_results[key].update({'video_id': video_id, 'frame_id': frame_id})
            combined_results[key]['score'] += weighted_score['image'] * score

    if buffer['ocr']:
        for video_id, frame_id, score in buffer['ocr']:
            key = f"{video_id}_{frame_id}"
            combined_results[key].update({'video_id': video_id, 'frame_id': frame_id})
            combined_results[key]['score'] += weighted_score['ocr'] * score

    if buffer['asr']:
        for video_id, frame_id, score in buffer['asr']:
            key = f"{video_id}_{frame_id}"
            combined_results[key].update({'video_id': video_id, 'frame_id': frame_id })
            combined_results[key]['score'] += weighted_score['asr'] * score
    
    if buffer['dislike_labels']:
        combined_results = {k: v for k, v in combined_results.items() if k not in buffer['dislike_labels']}

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
                'score': weighted_score['obj']
            }
            
    
    combined_results = sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)[:TOP_K]
    end_time = time.time()
    print("Total time for search: ", end_time - start_time)
    return combined_results
