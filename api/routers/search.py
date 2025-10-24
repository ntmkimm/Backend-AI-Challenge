from fastapi import APIRouter, HTTPException, Depends
from typing import List
from collections import defaultdict
import torch

from typing import List, Dict, Optional
from models.schemas import Query, ResultItem, InformationOfFrame, HistoryItem, ModelProvider
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

@router.post("/history", response_model=List[HistoryItem])
async def get_history(
    user_id: str = "anynomous",
    redis_service: RedisService = Depends(get_redis_service),
):
    hist = await redis_service.get_queries_history(user_id=user_id, limit=10)
    print("/history user_id: ", user_id)
    return [
        {"queries": [Query(**q) for q in item["queries"]]}
        for item in hist
    ]

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
    use_clip: bool = True,
    use_siglip2: bool = True,
    use_beit3: bool = True,
):
    model_provider = ModelProvider(clip=use_clip, beit3=use_beit3, siglip2=use_siglip2)
    print(f"/stage user_id: {user_id}")
    try:
        # Check valid stage_number
        if not 1 <= stage_number <= len(queries):
            raise HTTPException(status_code=400, detail="stage_number out of range")
        
        # queries = get_valid_queries(queries)  
        query = queries[stage_number - 1] # # Chọn đúng stage cần search
        if not get_valid_queries([query]): 
            raise HTTPException(status_code=404, detail="stage empty")
        
        results = await redis_service.get_one_answer_cached_redis(
            query=query,
            ttl_seconds=TIME_CACHE_ONE_QUERY,
            user_id=user_id,
            model_provider=model_provider,
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
    use_clip: bool = True,
    use_siglip2: bool = True,
    use_beit3: bool = True,
):
    model_provider = ModelProvider(clip=use_clip, beit3=use_beit3, siglip2=use_siglip2)
    print(f"/search user_id: {user_id}")
    try:
        if page * page_size > TOP_K: return []
        queries = get_valid_queries(queries=queries)
        redis_key = redis_service.make_tmp_search_result_key(user_id, queries, mode="normal", model_provider=model_provider)
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
                model_provider=model_provider,
            )

            start_time_algo = time.time()
            
            # Normalize scores per stage and collect results
            stage_results = []
            for stage_idx, hits in enumerate(all_answers):
                if not hits:
                    continue

                scores = [h["score"] for h in hits]
                min_s, max_s = min(scores), max(scores)
                norm_scores = [(s - min_s) / (max_s - min_s) if max_s != min_s else 1.0 for s in scores]

                stage_results.append([
                    (int(h["frame_id"]), h.get("video_id", ""), ns)
                    for h, ns in zip(hits, norm_scores)
                ])

            # Find common video_ids across all stages
            video_sets = [set(v for _, v, _ in stage) for stage in stage_results]
            common_videos = set.intersection(*video_sets) if video_sets else set()

            # Collect results only from common videos
            video_frames = defaultdict(list)  # video_id -> [(frame_id, score)]
            for stage in stage_results:
                for frame_id, video_id, score in stage:
                    if video_id in common_videos:
                        video_frames[video_id].append((frame_id, score))
            
            # Group frames within same video that are within 75 frames of each other
            FRAME_GROUP_THRESHOLD = 75
            grouped_results = []
            
            for video_id, frames in video_frames.items():
                # Sort frames by frame_id
                frames_sorted = sorted(set(frames), key=lambda x: x[0])
                
                if not frames_sorted:
                    continue
                
                # Union-Find / Connected Components approach
                groups = []
                current_group = [frames_sorted[0]]
                
                for i in range(1, len(frames_sorted)):
                    curr_frame_id, curr_score = frames_sorted[i]
                    
                    # Check if current frame can connect to any frame in current group
                    can_join = False
                    for group_frame_id, _ in current_group:
                        if abs(curr_frame_id - group_frame_id) <= FRAME_GROUP_THRESHOLD:
                            can_join = True
                            break
                    
                    if can_join:
                        current_group.append(frames_sorted[i])
                    else:
                        # Start new group
                        groups.append(current_group)
                        current_group = [frames_sorted[i]]
                
                # Don't forget the last group
                if current_group:
                    groups.append(current_group)
                
                # For each group, find the frame with highest score
                for group in groups:
                    max_score_frame = max(group, key=lambda x: x[1])
                    max_score = max_score_frame[1]
                    representative_frame_id = max_score_frame[0] # Get the frame_id of the highest score frame
                    
                    # Collect all frame_ids in this group
                    frame_ids = sorted([f[0] for f in group])
                    
                    # Store as (max_score, video_id, frame_ids_list, representative_frame_id)
                    grouped_results.append((max_score, video_id, frame_ids, representative_frame_id))

            # Sort by max score descending
            grouped_results.sort(key=lambda x: -x[0])
            all_results = grouped_results
            
            end_time = time.time()
            print("GROUPED TEMPORAL: ")
            print("Time for algorithm: ", end_time - start_time_algo)
            print("Tong thoi gian xu li: ", end_time - start_time)
            
            await redis_service.save_tmp_search_results_to_cache(
                redis_key=redis_key, 
                results=all_results, 
                ttl_seconds=TIME_CACHE_QUERIES
            )
        
        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        paged_results = all_results[start:end]
        
        # Format results - show only highest confidence frame per group
        # Store all grouped frame_ids in the id field for frontend reference
        return [
            ResultItem(
                id=f"{i + start}|{','.join(map(str, frame_ids))}",  # Encode grouped frames in ID
                videoId=video_id,
                confidence=round(score, 4),
                timestamp=str(representative_frame_id)  # Show the representative frame with the highest score
            )
            for i, (score, video_id, frame_ids, representative_frame_id) in enumerate(paged_results)
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
    use_clip: bool = True,
    use_siglip2: bool = True,
    use_beit3: bool = True,
):
    model_provider = ModelProvider(clip=use_clip, beit3=use_beit3, siglip2=use_siglip2)
    print(f"/chain_search user_id: {user_id}")
    try:
        if page * page_size > TOP_K: return []
        queries = get_valid_queries(queries=queries)
        redis_key = redis_service.make_tmp_search_result_key(user_id, queries, mode="chain", model_provider=model_provider)
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
                model_provider=model_provider,
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
                
                
                exist_chain = set()
                for idx, score in enumerate(dp_scores[-1]):
                    if (score.item() < 0): continue
                    flag = 1
                    for stage_i, path in enumerate(dp_paths[-1][idx]):
                        frame_id, frame_score = tensor_stages[stage_i][2][path]
                        if frame_id in exist_chain:
                            flag = 0
                            break 
                        exist_chain.add(frame_id)
                    for stage_i, path in enumerate(dp_paths[-1][idx]):
                        if not flag: break 
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
