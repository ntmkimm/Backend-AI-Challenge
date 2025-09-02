from fastapi import APIRouter, HTTPException, Depends
from typing import List
from collections import defaultdict
import torch

from typing import List, Dict, Optional
from models.schemas import Query, ResultItem, InformationOfFrame
from services.redis_service import RedisService
from services.polar_service import PolarService
from config.settings import MAX_FRAME_GAP, TOP_K, TIME_CACHE_ONE_QUERY, TIME_CACHE_QUERIES, MIN_FRAME_GAP
from dependencies.services import get_polar_service, get_redis_service
from core.utils import get_valid_queries
from utils.es_module import get_text_by_frame
from collections import defaultdict
import time
import numpy as np
import pickle
import asyncio

router = APIRouter(prefix="/embeddings") 

@router.post("/information", response_model=Optional[InformationOfFrame])
async def get_information(
    video_id: str, # L01_V001
    frame_id: str, # 10
    polar_service: PolarService = Depends(get_polar_service),
):
    try:
        frame_id = int(frame_id)
        es_data = await asyncio.to_thread(
            get_text_by_frame, video_id=video_id, frame_id=frame_id
        )

        pl_data = await asyncio.to_thread(
            polar_service.get_object_by_frame, video_id, frame_id
        )

        objects_str = ""
        if pl_data:
            objects_str = ", ".join(
                f"{k}={v}" for k, v in pl_data.items()
                if isinstance(v, (int, float)) and v > 0
            )

        if not es_data and not objects_str:
            return None

        return InformationOfFrame(
            ocr_text=es_data.get("ocr_text", "") if es_data else "",
            asr_text=es_data.get("asr_text", "") if es_data else "",
            objects=objects_str
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stage", response_model=List[ResultItem])
async def get_stage(
    queries: List[Query],
    stage_number: int = 1,   # stage_number tính từ 1 (client gửi lên)
    page: int = 1,
    page_size: int = 100,
    redis_service: RedisService = Depends(get_redis_service),
    user_id: str = 'anynomous',
):
    print(f"/stage user_id: {user_id}")
    try:
        # Check valid stage_number
        if not 1 <= stage_number <= len(queries):
            raise HTTPException(status_code=400, detail="stage_number out of range")
        queries = get_valid_queries(queries)  
        query = queries[stage_number - 1] # # Chọn đúng stage cần search
        
        results = await redis_service.get_one_answer_cached_redis(
            query=query,
            ttl_seconds=TIME_CACHE_ONE_QUERY,
            user_id=user_id
        ) or []

        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        paged_results = results[start:end]

        return [
            ResultItem(
                id=str(i + start),
                videoId=h.get("video_id", ""),
                confidence=round(h.get("score", 0), 4),
                timestamp=str(h.get("frame_id", ""))
            )
            for i, h in enumerate(paged_results)
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/search", response_model=List[ResultItem])
async def search_text(
    queries: List[Query],
    redis_service: RedisService = Depends(get_redis_service),
    page: int = 1,
    page_size: int = 100,
    user_id: str = 'anynomous',
):
    print(f"/search user_id: {user_id}")
    try:
        if page * page_size > TOP_K: return []
        queries = get_valid_queries(queries=queries)
        redis_key = redis_service.make_tmp_search_result_key(user_id, queries, mode="normal")
        cached_bytes = await redis_service.redis_client.get(redis_key)
        if cached_bytes is not None:
            all_results = pickle.loads(cached_bytes)
        else:
            await redis_service.flush_user_search_cache(user_id=user_id, mode="normal")
            start_time = time.time()
            all_answers = await redis_service.get_all_answers_cached_redis(
                queries=queries,
                ttl_seconds=TIME_CACHE_ONE_QUERY,
                user_id=user_id,
            )

            start_time_algo = time.time()
            device = "cuda"
            # Bước 1: Gộp tất cả hits của từng stage
            # all_answers là list (n_stage) của list (top_k dict)
            all_hits = []
            for stage_idx, hits in enumerate(all_answers):
                for h in hits:
                    all_hits.append((stage_idx, int(h["frame_id"]), h["score"], h.get("video_id", "")))

            # Bước 2: Gộp thành từng stage
            n_stage = len(queries)
            stage_to_hits = [[] for _ in range(n_stage)]
            for stage_idx, frame_id, score, video_id in all_hits:
                stage_to_hits[stage_idx].append((frame_id, score, video_id))
            
            # Bước 3: Tạo tensor cho toàn bộ các stage
            tensor_stages = []
            device = "cuda"
            for hits in stage_to_hits:
                if not hits:
                    tensor_stages.append((torch.tensor([], device=device), torch.tensor([], device=device), []))
                    continue
                stage_sorted = sorted(hits, key=lambda x: x[0])
                fids = torch.tensor([x[0] for x in stage_sorted], device=device)
                scores = torch.tensor([x[1] for x in stage_sorted], device=device)
                tensor_stages.append((fids, scores, stage_sorted))

            # Temporal scoring (giống cũ nhưng toàn bộ dataset)
            if len(tensor_stages[0][0]) == 0:
                return []  # Không có dữ liệu stage 0

            base_fids, base_scores, base_raw = tensor_stages[0]
            final_scores = base_scores.clone()
            for curr_fids, curr_scores, curr_raw in tensor_stages[1:]:
                if len(curr_fids) == 0:
                    continue
                curr_video_ids = np.array([x[2] for x in curr_raw])
                base_video_ids = np.array([x[2] for x in base_raw])
                # Tạo mask so sánh video_id
                video_mask = (curr_video_ids[:, None] == base_video_ids[None, :])
                diff = curr_fids[:, None] - base_fids[None, :]
                valid = (diff > 0) & (diff <= MAX_FRAME_GAP)
                # Dùng mask numpy convert thành torch để AND với valid
                video_mask_torch = torch.from_numpy(video_mask).to(valid.device)
                valid = valid & video_mask_torch
                decay = torch.sigmoid((MAX_FRAME_GAP / 2 - diff.float()) / 30)
                boost = curr_scores[:, None] * decay
                boost = torch.where(valid, boost, torch.zeros_like(boost))
                num_valid = valid.sum(dim=0).clamp(min=1)
                final_scores += boost.sum(dim=0) / num_valid

            final_results = []
            for i in range(len(base_fids)):
                frame_id, score, video_id = base_raw[i]  # just unpack the 3-tuple
                final_results.append((final_scores[i].item(), frame_id, video_id))

            # Format lại kết quả
            end_time = time.time()
            print("TEMPORAL: ")
            print("Time for algorithm: ", end_time - start_time_algo)
            print("Tong thoi gian xu li: ", end_time - start_time)
            final_results.sort(key=lambda x: -x[0])
            all_results = final_results  
            await redis_service.save_tmp_search_results_to_cache(redis_key=redis_key, results=all_results, ttl_seconds=TIME_CACHE_QUERIES)
        start = (page - 1) * page_size
        end = start + page_size
        paged_results = all_results[start:end]
        return [
            ResultItem(
                id=str(i + start),
                videoId=video_id,
                confidence=round(score, 4),
                timestamp=str(frame_id)
            )
            for i, (score, frame_id, video_id) in enumerate(paged_results)
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.post("/chain_search", response_model=List[ResultItem])
async def chain_search_text(
    queries: List[Query],
    redis_service: RedisService = Depends(get_redis_service),
    page: int = 1,
    page_size: int = 100,
    user_id: str = 'anonymous', 
):
    print(f"/chain_search user_id: {user_id}")
    try:
        if page * page_size > TOP_K: return []
        queries = get_valid_queries(queries=queries)
        redis_key = redis_service.make_tmp_search_result_key(user_id, queries, mode="chain")
        cached_bytes = await redis_service.redis_client.get(redis_key)
        if cached_bytes is not None:
            all_results = pickle.loads(cached_bytes)
        else:
            await redis_service.flush_user_search_cache(user_id=user_id, mode="chain")
            start_time = time.time()
            all_answers = await redis_service.get_all_answers_cached_redis(
                queries=queries,
                ttl_seconds=TIME_CACHE_ONE_QUERY,
                user_id=user_id,
            )
            
            start_time_algo = time.time()
            video_groups = defaultdict(lambda: [[] for _ in range(len(queries))])
            
            for stage_idx, hits in enumerate(all_answers):
                for h in hits:
                    video_groups[h["video_id"]][stage_idx].append(
                        (int(h["frame_id"]), h["score"])
                    )

            # === ALIGN AND SCORE CHAINS ===

            all_chains = []

            for vid, stage_hits in video_groups.items():
                if any(len(s) == 0 for s in stage_hits):
                    continue    

                tensor_stages = []
                for stage in stage_hits:
                    stage_sorted = sorted(stage)
                    fids = torch.tensor([f[0] for f in stage_sorted], device="cuda")
                    scores = torch.tensor([f[1] for f in stage_sorted], device="cuda")
                    tensor_stages.append((fids, scores, stage_sorted))

                n_stages = len(tensor_stages)
                dp_scores = [None] * n_stages
                dp_paths = [None] * n_stages

                dp_scores[0] = tensor_stages[0][1]
                dp_paths[0] = [[i] for i in range(len(tensor_stages[0][1]))]

                for i in range(1, n_stages):
                    prev_fids, prev_scores, _ = tensor_stages[i - 1]
                    curr_fids, curr_scores, _ = tensor_stages[i]

                    diff = curr_fids[:, None] - prev_fids[None, :]
                    # valid = (diff > MIN_FRAME_GAP) & (diff <= MAX_FRAME_GAP // len(queries))
                    valid = (diff > MIN_FRAME_GAP) & (diff <= MAX_FRAME_GAP)

                    decay = torch.sigmoid((MAX_FRAME_GAP / 2 - diff.float()) / 50)
                    temp_score = dp_scores[i - 1][None, :] + curr_scores[:, None] * decay
                    temp_score = torch.where(valid, temp_score, torch.full_like(temp_score, -1e9))

                    max_vals, max_idxs = temp_score.max(dim=1)
                    dp_scores[i] = max_vals
                    dp_paths[i] = [dp_paths[i - 1][j.item()] + [k] for j, k in zip(max_idxs, range(len(curr_fids)))]
                
                

                for idx, score in enumerate(dp_scores[-1]):
                    if (score.item() < 0): continue
                    for stage_i, path in enumerate(dp_paths[-1][idx]):
                        all_chains.append((score.item(), tensor_stages[stage_i][2][path], vid))

            # Sort chains across all videos
            all_chains.sort(key=lambda x: -x[0])

            all_results = all_chains
            # Cache toàn bộ results dạng tuple (score, filepath, frame_id, video_id, stage)
            await redis_service.save_tmp_search_results_to_cache(redis_key=redis_key, results=all_results, ttl_seconds=TIME_CACHE_QUERIES)
            end_time = time.time()
            print("CHAIN: ")
            print("Time for algorithm: ", end_time - start_time_algo)
            print("Tong thoi gian xu li: ", end_time - start_time)

        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        paged_results = all_results[start:end]
        # Convert ra ResultItem
        result_list = []
        for i, (score, frameinfo, video_id) in enumerate(paged_results):
            frame_id, frame_score = frameinfo
            stage = i % len(queries)
            result_list.append(
                ResultItem(
                    id=f"{i+start}",
                    videoId=video_id,
                    confidence=round(score, 4),
                    timestamp=str(frame_id)
                )
            )
        return result_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
