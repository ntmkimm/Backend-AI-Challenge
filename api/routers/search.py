from fastapi import APIRouter, HTTPException, Depends
from typing import List
from collections import defaultdict
import torch
import heapq

from models.schemas import Query, ResultItem, OCRSearchRequest, NearbyFramesRequest, ImageSearchRequest, ImageSearchResult
from services.clip_service import CLIPService
from services.milvus_service import MilvusService
from services.redis_service import RedisService
from services.polar_service import PolarService
from config.settings import MAX_FRAME_GAP, MEDIA_SERVER_URL, TOP_K
from dependencies.services import get_clip_service, get_milvus_service, get_polar_service, get_redis_service
from core.utils import get_valid_queries
from collections import defaultdict
import time
import concurrent.futures
import numpy as np
import pickle
import asyncio

router = APIRouter(prefix="/embeddings")

@router.post("/stage", response_model=List[ResultItem])
async def get_stage(
    queries: List[Query],
    stage_number: int = 1,   # stage_number tính từ 1 (client gửi lên)
    page: int = 1,
    page_size: int = 100,
    clip_service: CLIPService = Depends(get_clip_service),
    milvus_service: MilvusService = Depends(get_milvus_service),
    polar_service: PolarService = Depends(get_polar_service),
    redis_service: RedisService = Depends(get_redis_service),
):
    try:
        # Check valid stage_number
        if not 1 <= stage_number <= len(queries):
            raise HTTPException(status_code=400, detail="stage_number out of range")
        query = queries[stage_number - 1]  # Chọn đúng stage cần search

        results = await redis_service.get_one_answer_cached_redis(
            query=query,
            clip_service=clip_service,
            milvus_service=milvus_service,
            polar_service=polar_service,
            ttl_seconds=3600,
        ) or []

        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        paged_results = results[start:end]

        return [
            ResultItem(
                id=str(i + start),
                videoId=h.get("video_id", ""),
                title=f"{h.get('video_id', '')}/{h.get('frame_id', '')}",
                thumbnail=f"{MEDIA_SERVER_URL}/{h.get('filepath', '')}",
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
    clip_service: CLIPService = Depends(get_clip_service),
    milvus_service: MilvusService = Depends(get_milvus_service),
    polar_service: PolarService = Depends(get_polar_service),
    redis_service: RedisService = Depends(get_redis_service),
    page: int = 1,
    page_size: int = 100,
    user_id: str = 'anynomous',
):
    try:
        if page * page_size >= TOP_K: return []
        redis_key = redis_service.make_tmp_search_result_key(user_id, queries, mode="normal")
        cached_bytes = await asyncio.to_thread(redis_service.redis_client.get, redis_key)
        if cached_bytes is not None:
            all_results = pickle.loads(cached_bytes)
        else:
            start_time = time.time()
            queries = get_valid_queries(queries=queries)
            all_answers = await redis_service.get_all_answers_cached_redis(
                queries=queries,
                clip_service=clip_service,
                milvus_service=milvus_service,
                polar_service=polar_service,
                ttl_seconds=3600 
            )
            start_time_algo = time.time()
            device = clip_service.device
            # Bước 1: Gộp tất cả hits của từng stage
            # all_answers là list (n_stage) của list (top_k dict)
            all_hits = []
            for stage_idx, hits in enumerate(all_answers):
                for h in hits:
                    all_hits.append((stage_idx, int(h["frame_id"]), h["score"], h["filepath"], h.get("video_id", "")))

            # Bước 2: Gộp thành từng stage
            n_stage = len(queries)
            stage_to_hits = [[] for _ in range(n_stage)]
            for stage_idx, frame_id, score, path, video_id in all_hits:
                stage_to_hits[stage_idx].append((frame_id, score, path, video_id))
            
            # Bước 3: Tạo tensor cho toàn bộ các stage
            tensor_stages = []
            device = clip_service.device
            for hits in stage_to_hits:
                if not hits:
                    tensor_stages.append((torch.tensor([], device=device), torch.tensor([], device=device), []))
                    continue
                stage_sorted = sorted(hits)
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
                curr_video_ids = np.array([x[3] for x in curr_raw])
                base_video_ids = np.array([x[3] for x in base_raw])
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
                frame_id, score, path, video_id = (*base_raw[i][:3], base_raw[i][3])
                final_results.append((final_scores[i].item(), path, frame_id, video_id))

            # Format lại kết quả
            end_time = time.time()
            print("TEMPORAL: ")
            print("Time for algorithm: ", end_time - start_time_algo)
            print("Tong thoi gian xu li: ", end_time - start_time)
            final_results.sort(key=lambda x: -x[0])
            all_results = final_results  
            await redis_service.save_tmp_search_results_to_cache(redis_key=redis_key, results=all_results, ttl_seconds=300)
        start = (page - 1) * page_size
        end = start + page_size
        paged_results = all_results[start:end]
        return [
            ResultItem(
                id=str(i + start),
                videoId=video_id,
                title=f"{video_id}/{frame_id}",
                thumbnail=f"{MEDIA_SERVER_URL}/{path}",
                confidence=round(score, 4),
                timestamp=str(frame_id)
            )
            for i, (score, path, frame_id, video_id) in enumerate(paged_results)
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.post("/chain_search", response_model=List[ResultItem])
async def chain_search_text(
    queries: List[Query],
    clip_service: CLIPService = Depends(get_clip_service),
    milvus_service: MilvusService = Depends(get_milvus_service),
    polar_service: PolarService = Depends(get_polar_service),
    redis_service: RedisService = Depends(get_redis_service),
    page: int = 1,
    page_size: int = 100,
    user_id: str = 'anonymous', 
):
    try:
        if page * page_size >= TOP_K: return []
        redis_key = redis_service.make_tmp_search_result_key(user_id, queries, mode="chain")
        cached_bytes = await asyncio.to_thread(redis_service.redis_client.get, redis_key)
        if cached_bytes is not None:
            all_results = pickle.loads(cached_bytes)
        else:
            start_time = time.time()
            queries = get_valid_queries(queries=queries)
            all_answers = await redis_service.get_all_answers_cached_redis(
                queries=queries,
                clip_service=clip_service,
                milvus_service=milvus_service,
                polar_service=polar_service,
                ttl_seconds=3600
            )
            start_time_algo = time.time()
            device = clip_service.device
            tensor_stages = []
            for hits in all_answers:
                if hits:
                    hits_sorted = sorted(hits, key=lambda h: int(h["frame_id"]))
                    fids = torch.tensor([int(h["frame_id"]) for h in hits_sorted], device=device)
                    scores = torch.tensor([h["score"] for h in hits_sorted], device=device)
                    paths = [h["filepath"] for h in hits_sorted]
                    vids = [h.get("video_id", "") for h in hits_sorted]
                    tensor_stages.append((fids, scores, paths, vids))
                else:
                    tensor_stages.append((
                        torch.tensor([], device=device),
                        torch.tensor([], device=device),
                        [],
                        []
                    ))

            if any(len(stage[0]) == 0 for stage in tensor_stages):
                return []

            n_stages = len(tensor_stages)
            dp_scores = [None] * n_stages
            dp_paths = [None] * n_stages

            dp_scores[0] = tensor_stages[0][1]
            dp_paths[0] = [[i] for i in range(len(tensor_stages[0][1]))]

            for i in range(1, n_stages):
                prev_fids, prev_scores, _, prev_vids = tensor_stages[i - 1]
                curr_fids, curr_scores, _, curr_vids = tensor_stages[i]

                prev_vids_arr = np.array(prev_vids)
                curr_vids_arr = np.array(curr_vids)
                video_mask = (curr_vids_arr[:, None] == prev_vids_arr[None, :])
                diff = curr_fids[:, None] - prev_fids[None, :]
                valid = (diff > 0) & (diff <= MAX_FRAME_GAP // n_stages)
                video_mask_torch = torch.from_numpy(video_mask).to(device)
                valid = valid & video_mask_torch

                decay = torch.sigmoid((MAX_FRAME_GAP / 2 - diff.float()) / 50)
                temp_score = dp_scores[i - 1][None, :] + curr_scores[:, None] * decay
                temp_score = torch.where(valid, temp_score, torch.full_like(temp_score, -1e9))

                max_vals, max_idxs = temp_score.max(dim=1)
                dp_scores[i] = max_vals
                dp_paths[i] = [dp_paths[i - 1][j.item()] + [k] for j, k in zip(max_idxs, range(len(curr_fids)))]

            # Trích xuất chain tốt nhất cuối cùng
            all_chains = []
            for idx, score in enumerate(dp_scores[-1]):
                path_indices = dp_paths[-1][idx]
                chain = []
                for stage_idx, item_idx in enumerate(path_indices):
                    frame_id = tensor_stages[stage_idx][0][item_idx].item()
                    score_ = tensor_stages[stage_idx][1][item_idx].item()
                    path = tensor_stages[stage_idx][2][item_idx]
                    video_id = tensor_stages[stage_idx][3][item_idx]
                    chain.append((frame_id, score_, path, video_id))
                all_chains.append((score.item(), chain))

            # Sort chains theo score giảm dần
            all_chains.sort(key=lambda x: -x[0])

            # Convert ra dạng list kết quả phẳng
            results = []
            for i, (score, chain) in enumerate(all_chains):
                for stage, (frame_id, distance, filepath, video_id) in enumerate(chain):
                    results.append((
                        score, filepath, frame_id, video_id, stage
                    ))

            all_results = results
            # Cache toàn bộ results dạng tuple (score, filepath, frame_id, video_id, stage)
            await redis_service.save_tmp_search_results_to_cache(redis_key=redis_key, results=all_results, ttl_seconds=300)
            end_time = time.time()
            print("CHAIN: ")
            print("Time for algorithm: ", end_time - start_time_algo)
            print("Tong thoi gian xu li: ", end_time - start_time)

        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        paged_results = all_results[start:end]
        # Convert ra ResultItem
        return [
            ResultItem(
                id=f"{i+start}",
                videoId=video_id,
                title=f"{video_id}/{frame_id}-stage{stage}-{round(score, 2)}",
                thumbnail=f"{MEDIA_SERVER_URL}/{filepath}",
                confidence=round(score, 4),
                timestamp=str(frame_id)
            )
            for i, (score, filepath, frame_id, video_id, stage) in enumerate(paged_results)
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/nearby_frames", response_model=List[ResultItem])
async def get_nearby_frames(
    request: NearbyFramesRequest,
    milvus_service: MilvusService = Depends(get_milvus_service)
):
    try:
        frames = milvus_service.get_frames_by_video_id(request.video_id)
        frames = sorted(frames, key=lambda x: int(x['frame_id']))
        
        # Find target frame index
        target_idx = next(i for i, frame in enumerate(frames) 
                         if int(frame['frame_id']) == request.frame_id)
        
        # Get window_size frames before and after
        start_idx = max(0, target_idx - request.window_size)
        end_idx = min(len(frames), target_idx + request.window_size + 1)
        nearby_frames = frames[start_idx:end_idx]
        
        return [
            ResultItem(
                id=str(i),
                videoId=frame['video_id'],
                title=f"{frame['video_id']}/{frame['frame_id']}",
                thumbnail=f"{MEDIA_SERVER_URL}/{frame['filepath']}",
                confidence=1.0 if int(frame['frame_id']) == request.frame_id else 0.9,
                timestamp=str(frame['frame_id'])
            )
            for i, frame in enumerate(nearby_frames)
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
