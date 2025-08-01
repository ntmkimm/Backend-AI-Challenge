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

router = APIRouter(prefix="/embeddings")

@router.post("/search", response_model=List[ResultItem])
async def search_text(
    queries: List[Query],
    clip_service: CLIPService = Depends(get_clip_service),
    milvus_service: MilvusService = Depends(get_milvus_service),
    polar_service: PolarService = Depends(get_polar_service),
    redis_service: RedisService = Depends(get_redis_service),
):
    try:
        start_time = time.time()
        queries = get_valid_queries(queries=queries)
        print('before search')
        all_answers = redis_service.get_all_answers_cached_redis(
            queries=queries,
            clip_service=clip_service,
            milvus_service=milvus_service,
            polar_service=polar_service,
            ttl_seconds = 3600 
        )
        # Group by video ID
        print('before rerank')
        video_groups = defaultdict(lambda: [[] for _ in range(len(queries))])
        for stage_idx, hits in enumerate(all_answers):
            for h in hits:
                video_groups[h["video_id"]][stage_idx].append(
                    (int(h["frame_id"]), h["score"], h["filepath"])
                )

        # Temporal scoring
        final_results = []
        for vid, stage_hits in video_groups.items():
            if any(len(stage) == 0 for stage in stage_hits):
                continue

            tensor_stages = []
            for stage in stage_hits:
                stage_sorted = sorted(stage)
                fids = torch.tensor([x[0] for x in stage_sorted], device=clip_service.device)
                scores = torch.tensor([x[1] for x in stage_sorted], device=clip_service.device)
                tensor_stages.append((fids, scores, stage_sorted))

            base_fids, base_scores, base_raw = tensor_stages[0]
            final_scores = base_scores.clone()

            for curr_fids, curr_scores, _ in tensor_stages[1:]:
                diff = curr_fids[:, None] - base_fids[None, :]
                valid = (diff > 0) & (diff <= MAX_FRAME_GAP // len(queries))
                decay = torch.sigmoid((MAX_FRAME_GAP / 2 - diff.float()) / 30)
                boost = curr_scores[:, None] * decay
                boost = torch.where(valid, boost, torch.zeros_like(boost))
                num_valid = valid.sum(dim=0).clamp(min=1)
                final_scores += boost.sum(dim=0) / num_valid

            for i in range(len(base_fids)):
                frame_id, dist, path = base_raw[i]
                final_results.append((final_scores[i].item(), path, frame_id, vid))

        # Format results
        end_time = time.time()
        print("Thoi gian xu li: ", end_time - start_time)
        final_results.sort(key=lambda x: -x[0])
        return [
            ResultItem(
                id=str(i),
                videoId=video_id,
                title=f"{video_id}/{frame_id}",
                thumbnail=f"{MEDIA_SERVER_URL}/{path}",
                confidence=round(score, 4),
                timestamp=str(frame_id)
            )
            for i, (score, path, frame_id, video_id) in enumerate(final_results)
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
):
    try:
        queries = get_valid_queries(queries=queries)
        all_answers = redis_service.get_all_answers_cached_redis(
            queries=queries,
            clip_service=clip_service,
            milvus_service=milvus_service,
            polar_service=polar_service,
            ttl_seconds = 3600 
        )
        # Group by video ID
        video_groups = defaultdict(lambda: [[] for _ in range(len(queries))])
        for stage_idx, hits in enumerate(all_answers):
            for h in hits:
                video_groups[h["video_id"]][stage_idx].append(
                    (int(h["frame_id"]), h["score"], h["filepath"])
                )

        all_chains = []

        for vid, stage_hits in video_groups.items():
            if any(len(s) == 0 for s in stage_hits):
                continue

            tensor_stages = []
            for stage in stage_hits:
                stage_sorted = sorted(stage)
                fids = torch.tensor([f[0] for f in stage_sorted], device=clip_service.device)
                scores = torch.tensor([f[1] for f in stage_sorted], device=clip_service.device)
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
                valid = (diff > 0) & (diff <= MAX_FRAME_GAP)

                # decay = (MAX_FRAME_GAP - diff) / MAX_FRAME_GAP
                decay = torch.sigmoid((MAX_FRAME_GAP / 2 - diff.float()) / 50)
                temp_score = dp_scores[i - 1][None, :] + curr_scores[:, None] * decay
                temp_score = torch.where(valid, temp_score, torch.full_like(temp_score, -1e9))

                max_vals, max_idxs = temp_score.max(dim=1)
                dp_scores[i] = max_vals
                dp_paths[i] = [dp_paths[i - 1][j.item()] + [k] for j, k in zip(max_idxs, range(len(curr_fids)))]

            for idx, score in enumerate(dp_scores[-1]):
                for stage_i, path in enumerate(dp_paths[-1][idx]):
                    all_chains.append((score.item(), tensor_stages[stage_i][2][path], vid))

        # Sort chains across all videos
        all_chains.sort(key=lambda x: -x[0])
        return [
            ResultItem(
                id=str(i),
                videoId=video_id,
                title=f"{video_id}/{frame_id}-{i % len(queries)}-{round(score, 2)}",
                thumbnail=f"{MEDIA_SERVER_URL}/{path}",
                confidence=round(score, 4),
                timestamp=str(frame_id)
            )
            for i, (score, (frame_id, distance_score, path), video_id) in enumerate(all_chains)
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
