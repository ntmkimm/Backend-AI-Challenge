from fastapi import APIRouter, HTTPException, Depends
from typing import List, Set
from collections import defaultdict
import torch

from typing import List, Dict, Optional
from models.schemas import Query, ResultItem, InformationOfFrame, HistoryItem, ModelProvider
from services.redis_service import RedisService
from services.polar_service import PolarService
from services.interval_service import IntervalService
from config.settings import MAX_FRAME_GAP, TOP_K, TIME_CACHE_ONE_QUERY, TIME_CACHE_QUERIES, MIN_FRAME_GAP
from dependencies.services import get_polar_service, get_redis_service, get_interval_service
from core.utils import get_valid_queries
from utils.es_module import get_text_by_frame
from collections import defaultdict
import time
import numpy as np
import pickle
import asyncio
from models.schemas import ReverseObjectFilterResponse 
from models.schemas import DDGQuery,DDGResult,DDGImageQuery,DDGImageResult
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from duckduckgo_search import DDGS

router = APIRouter(prefix="/embeddings") 




# ---------- WRAPPER (no LLM needed) ----------
# You can override defaults at request time; these are just sensible fallbacks.
_ddg_wrapper_default = DuckDuckGoSearchAPIWrapper(
    region="wt-wt",        # worldwide
    safesearch="moderate", # 'off' | 'moderate' | 'strict'
    time="",               # '', 'd', 'w', 'm', 'y'
    max_results=5
)

# ---------- ENDPOINT ----------
@router.post("/duckduckgo/search", response_model=List[DDGResult])
async def duckduckgo_search(payload: DDGQuery):
    """
    Simple DuckDuckGo search via LangChain utilities.
    No LLM is used. Returns a list of {title, link, snippet}.
    """
    try:
        # Create a per-request wrapper (so caller can override params)
        ddg = DuckDuckGoSearchAPIWrapper(
            region=payload.region,
            safesearch=payload.safesearch,
            time=payload.time,
            max_results=payload.max_results,
        )

        # ddg.results(...) is blocking; run it in a thread to keep this endpoint async.
        def _search():
            # Returns List[{"title": str, "link": str, "snippet": str}]
            return ddg.results(payload.query, payload.max_results)

        raw_results = await asyncio.to_thread(_search)

        # Normalize/validate to DDGResult
        results: List[DDGResult] = []
        for r in raw_results or []:
            results.append(DDGResult(
                title=r.get("title", ""),
                link=r.get("link", ""),
                snippet=r.get("snippet", ""),
            ))
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DuckDuckGo search failed: {e}")
# ---------- NEW IMAGE SEARCH ENDPOINT ----------
@router.post("/duckduckgo/images", response_model=List[DDGImageResult])
async def duckduckgo_image_search(payload: DDGImageQuery):
    """
    Performs a DuckDuckGo image search using the 'duckduckgo-search' library.
    This method scrapes results and is not an official API.
    Returns a list of image result objects.
    """
    try:
        # The 'duckduckgo-search' library's functions are synchronous (blocking).
        # We must run them in a separate thread to keep our FastAPI endpoint async.
        def _search_images():
            # Using a context manager is recommended by the library's author
            with DDGS() as ddgs:
                # The images() function returns a generator, so we convert it to a list
                results = list(ddgs.images(
                    keywords=payload.query,
                    region=payload.region,
                    safesearch=payload.safesearch,
                    max_results=payload.max_results,
                ))
            return results

        # Run the blocking search function in a thread pool
        raw_results = await asyncio.to_thread(_search_images)

        # Normalize the dictionary results into our Pydantic response model.
        # This ensures the response is always in the format we expect.
        results: List[DDGImageResult] = []
        for r in raw_results or []:
            # The library returns keys that match our Pydantic model,
            # so we can unpack the dictionary directly.
            results.append(DDGImageResult(**r))
        
        return results

    except Exception as e:
        # If anything goes wrong during the search/scraping, return a server error.
        raise HTTPException(status_code=500, detail=f"DuckDuckGo image search failed: {e}")
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
# Add this to your existing router file (after the existing endpoints)

@router.post("/objects/filter_reverse_lookup", response_model=ReverseObjectFilterResponse)
async def get_objects_from_results_reverse_lookup(
    results: List[ResultItem],
    polar_service: PolarService = Depends(get_polar_service),
):
    """
    Extract all unique objects from a list of result items and
    return a mapping of object name to a list of frame keys where it appears,
    along with global counts.
    """
    try:
        global_object_counts = defaultdict(int)
        # New structure: map object name to a set of frame keys (use set to avoid duplicates)
        objects_to_frame_keys_map: Dict[str, Set[str]] = defaultdict(set)

        unique_frame_keys: Set[str] = set()
        frame_key_to_item_map: Dict[str, ResultItem] = {} # Stored if needed, not strictly for this schema

        for item in results:
            video_id = item.videoId
            representative_frame_id_str = item.timestamp.split('|')[0]
            frame_key = f"{video_id}-{representative_frame_id_str}"
            unique_frame_keys.add(frame_key)
            frame_key_to_item_map[frame_key] = item

        # Batch fetch objects for all unique frame keys
        tasks = []
        frame_keys_in_order = [] # To maintain order of results
        for frame_key in unique_frame_keys:
            video_id, frame_id_str = frame_key.split('-')
            frame_id = int(frame_id_str)
            tasks.append(
                asyncio.to_thread(polar_service.get_object_by_frame, video_id, frame_id)
            )
            frame_keys_in_order.append(frame_key) # Keep original order

        all_pl_data = await asyncio.gather(*tasks)

        for i, pl_data in enumerate(all_pl_data):
            frame_key = frame_keys_in_order[i] # Get the original frame_key for this result

            if pl_data:
                for obj_name, count in pl_data.items():
                    if isinstance(count, (int, float)) and count > 0:
                        global_object_counts[obj_name] += int(count)
                        objects_to_frame_keys_map[obj_name].add(frame_key) # Add frame_key to the set

        # Convert sets to lists for the final response model
        final_objects_to_frame_keys: Dict[str, List[str]] = {
            obj: sorted(list(frame_keys)) # Sort frame keys for consistent output
            for obj, frame_keys in objects_to_frame_keys_map.items()
        }

        # Sort global object counts by count (descending) as before
        sorted_global_object_counts = dict(sorted(global_object_counts.items(), key=lambda x: -x[1]))

        return ReverseObjectFilterResponse(
            objects_to_frame_keys=final_objects_to_frame_keys,
            global_object_counts=sorted_global_object_counts
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
                # Avoid division by zero if all scores are the same
                min_s, max_s = min(scores), max(scores)
                norm_scores = [(s - min_s) / (max_s - min_s) if max_s != min_s else 1.0 for s in scores]

                stage_results.append([
                    (int(h["frame_id"]), h.get("video_id", ""), ns)
                    for h, ns in zip(hits, norm_scores)
                ])

            # Find common video_ids across all stages
            # If stage_results is empty, common_videos should also be empty
            common_videos = set.intersection(*(set(v for _, v, _ in stage) for stage in stage_results)) if stage_results else set()

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
                # Sort frames by frame_id for efficient grouping
                frames_sorted = sorted(frames, key=lambda x: x[0])
                
                if not frames_sorted:
                    continue
                
                # Optimized Grouping: Iterate and merge
                current_group = []
                for frame_id, score in frames_sorted:
                    if not current_group:
                        current_group.append((frame_id, score))
                    else:
                        # Check if the current frame is close enough to any frame in the current group
                        # We only need to check against the last frame added to the group if sorted
                        # or more robustly, against all frames if the group isn't necessarily contiguous,
                        # but given sorted input, checking against the last element's max/min range is usually enough.
                        # For simplicity and robustness, let's stick to the existing "any frame in group" logic,
                        # but with sorted data, we can be more efficient.
                        
                        # Find the min and max frame_id in the current_group
                        min_group_frame_id = min(f[0] for f in current_group)
                        max_group_frame_id = max(f[0] for f in current_group)

                        if abs(frame_id - min_group_frame_id) <= FRAME_GROUP_THRESHOLD or \
                           abs(frame_id - max_group_frame_id) <= FRAME_GROUP_THRESHOLD:
                            current_group.append((frame_id, score))
                        else:
                            # Current frame is too far, finalize current group and start a new one
                            # Find the frame with the highest score in the finished group
                            max_score_frame = max(current_group, key=lambda x: x[1])
                            grouped_results.append((max_score_frame[1], video_id, max_score_frame[0])) # (score, video_id, frame_id)
                            current_group = [(frame_id, score)] # Start new group
                
                # Add the last group if it's not empty
                if current_group:
                    max_score_frame = max(current_group, key=lambda x: x[1])
                    grouped_results.append((max_score_frame[1], video_id, max_score_frame[0]))

            # Sort by max score descending
            grouped_results.sort(key=lambda x: -x[0])
            all_results = grouped_results # Now all_results is (score, video_id, frame_id)

            end_time = time.time()
            print("GROUPED TEMPORAL (Optimized): ")
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
        
        # Format results - simplified to show only the highest confidence frame per group
        return [
            ResultItem(
                id=f"{i + start}", # Simple ID for the result item
                videoId=video_id,
                confidence=round(score, 4),
                timestamp=str(frame_id) # Only the representative frame_id
            )
            for i, (score, video_id, frame_id) in enumerate(paged_results)
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/chain_search", response_model=List[ResultItem])
async def chain_search_text(
    queries: List[Query],
    redis_service: RedisService = Depends(get_redis_service),
    interval_service: IntervalService = Depends(get_interval_service),
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
                    s = score.item()
                    if s < 0:
                        continue

                    paths = dp_paths[-1][idx]
                    frames = [tensor_stages[stage_i][2][path] for stage_i, path in enumerate(paths)]
                    frame_ids = [f[0] for f in frames]

                    # Cho phép frame trùng nếu nó thuộc stage_i = 0
                    duplicated = any(fid in exist_chain for fid in frame_ids[:])
                    if duplicated:
                        continue

                    # Cập nhật: chỉ thêm các frame từ stage_i > 0 vào exist_chain
                    exist_chain.update(frame_ids[:])

                    # Kiểm tra interval
                    interval = interval_service.get_interval(vid, frame_ids[0])
                    if interval[0] or interval[1]:
                        inside = all(
                            ((interval[0] is None or fid >= interval[0]) and
                            (interval[1] is None or fid <= interval[1]))
                            for fid in frame_ids
                        )
                        if not inside:
                            continue

                    # Nếu qua hết các bước, thêm vào kết quả
                    all_chains.extend((s, f, vid) for f in frames)
                    
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
            frame_id, _ = frameinfo
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
