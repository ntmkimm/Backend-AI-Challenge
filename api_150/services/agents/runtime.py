from __future__ import annotations

from typing import List, Optional
import asyncio
import pickle
import time
import numpy as np
import torch
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate

from models.schemas import Query, ModelProvider, ResultItem, InformationOfFrame
from services.polar_service import PolarService
from config.settings import (
    MAX_FRAME_GAP,
    MIN_FRAME_GAP,
    TOP_K,
    TIME_CACHE_ONE_QUERY,
    TIME_CACHE_QUERIES,
)
from core.utils import get_valid_queries
from utils.es_module import get_text_by_frame

# Create DuckDuckGo wrappers for reuse (thread-safe)
_DDG_IMAGE_WRAPPER = DuckDuckGoSearchAPIWrapper(
    region="vn-vn",
    max_results=50,
    source="images"
)

_DDG_WEB_WRAPPER = DuckDuckGoSearchAPIWrapper(
    region="vn-vn",
    max_results=20,
    source="text"
)


class AgentRuntime:
    """
    In-process runtime that implements the same business logic as your FastAPI endpoints.
    Both the agent and the routers can call this class to avoid duplication & HTTP overhead.
    """

    def __init__(self, redis_service, polar_service: Optional[PolarService] = None):
        self.redis = redis_service
        self.polar = polar_service

    # ---------- /embeddings/search ----------
    async def search(
        self,
        queries: List[Query],
        provider: ModelProvider,
        user_id: str = "agent",
        page: int = 1,
        page_size: int = 100,
    ) -> List[ResultItem]:
        """
        Replicates your /search endpoint temporally-boosted scoring logic.
        Returns a list[ResultItem].
        """
        if page * page_size > TOP_K:
            return []

        queries = get_valid_queries(queries)
        redis_key = self.redis.make_tmp_search_result_key(
            user_id=user_id, queries=queries, mode="normal", model_provider=provider
        )

        cached_bytes = await self.redis.redis_client.get(redis_key)
        if cached_bytes is not None:
            all_results = pickle.loads(cached_bytes)
        else:
            await self.redis.flush_user_search_cache(user_id=user_id, mode="normal")
            
            start_time = time.time()

            all_answers = await self.redis.get_all_answers_cached_redis(
                queries=queries,
                ttl_seconds=TIME_CACHE_ONE_QUERY,
                user_id=user_id,
                model_provider=provider,
            )

            start_algo = time.time()

            # --- Stage hits flatten ---
            all_hits = []
            for stage_idx, hits in enumerate(all_answers):
                if not hits:
                    continue
                for h in hits:
                    all_hits.append(
                        (stage_idx, int(h["frame_id"]), float(h["score"]), h.get("video_id", ""))
                    )

            # --- Group by stage ---
            n_stage = len(queries)
            stage_to_hits = [[] for _ in range(n_stage)]
            for stage_idx, frame_id, score, vid in all_hits:
                stage_to_hits[stage_idx].append((frame_id, score, vid))

            # --- Build tensors per stage ---
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tensor_stages = []
            for hits in stage_to_hits:
                if not hits:
                    tensor_stages.append(
                        (torch.tensor([], device=device), torch.tensor([], device=device), [])
                    )
                    continue
                stage_sorted = sorted(hits, key=lambda x: x[0])
                fids = torch.tensor([x[0] for x in stage_sorted], device=device)
                scores = torch.tensor([x[1] for x in stage_sorted], device=device)
                tensor_stages.append((fids, scores, stage_sorted))

            # --- Temporal boosting ---
            if len(tensor_stages[0][0]) == 0:
                return []

            base_fids, base_scores, base_raw = tensor_stages[0]
            final_scores = base_scores.clone()

            for curr_fids, curr_scores, curr_raw in tensor_stages[1:]:
                if len(curr_fids) == 0:
                    continue
                curr_vids = np.array([x[2] for x in curr_raw])
                base_vids = np.array([x[2] for x in base_raw])

                video_mask = (curr_vids[:, None] == base_vids[None, :])
                diff = curr_fids[:, None] - base_fids[None, :]
                valid = (diff > 0) & (diff <= MAX_FRAME_GAP)

                video_mask_torch = torch.from_numpy(video_mask).to(valid.device)
                valid = valid & video_mask_torch

                decay = torch.sigmoid((MAX_FRAME_GAP / 2 - diff.float()) / 30)
                boost = torch.where(valid, curr_scores[:, None] * decay, torch.zeros_like(decay))
                num_valid = valid.sum(dim=0).clamp(min=1)
                final_scores += boost.sum(dim=0) / num_valid

            final_results = []
            for i in range(len(base_fids)):
                frame_id, _score, vid = base_raw[i]
                final_results.append((float(final_scores[i].item()), int(frame_id), vid))

            final_results.sort(key=lambda x: -x[0])
            all_results = final_results

            end_time = time.time()
            print("TEMPORAL:")
            print("  algorithm_time =", end_time - start_algo)
            print("  total_time     =", end_time - start_time)

            await self.redis.save_tmp_search_results_to_cache(
                redis_key=redis_key, results=all_results, ttl_seconds=TIME_CACHE_QUERIES
            )

        # Pagination
        start = (page - 1) * page_size
        end = start + page_size
        paged = all_results[start:end]

        return [
            ResultItem(
                id=str(i + start),
                videoId=vid,
                confidence=round(score, 4),
                timestamp=str(fid),
            )
            for i, (score, fid, vid) in enumerate(paged)
        ]

    # ---------- /embeddings/chain_search ----------
    async def chain_search(
        self,
        queries: List[Query],
        provider: ModelProvider,
        user_id: str = "agent",
        page: int = 1,
        page_size: int = 100,
    ) -> List[ResultItem]:
        """
        Mirror the FastAPI /chain_search DP alignment logic.
        """
        if page * page_size > TOP_K:
            return []
    
        queries = get_valid_queries(queries)
        redis_key = self.redis.make_tmp_search_result_key(
            user_id=user_id, queries=queries, mode="chain", model_provider=provider
        )
    
        cached_bytes = await self.redis.redis_client.get(redis_key)
        if cached_bytes is not None:
            all_results = pickle.loads(cached_bytes)
        else:
            await self.redis.flush_user_search_cache(user_id=user_id, mode="chain")
            t0 = time.time()
    
            all_answers = await self.redis.get_all_answers_cached_redis(
                queries=queries,
                ttl_seconds=TIME_CACHE_ONE_QUERY,
                user_id=user_id,
                model_provider=provider,
            )
    
            t_algo = time.time()
            from collections import defaultdict as _defaultdict
    
            video_groups = _defaultdict(lambda: [[] for _ in range(len(queries))])
            for stage_idx, hits in enumerate(all_answers):
                if not hits:
                    continue
                for h in hits:
                    vid = h["video_id"]
                    fid = int(h["frame_id"])
                    sc = float(h["score"])
                    video_groups[vid][stage_idx].append((fid, sc))
    
            all_chains = []
            device = "cuda" if torch.cuda.is_available() else "cpu"
    
            for vid, stage_hits in video_groups.items():
                if any(len(s) == 0 for s in stage_hits):
                    continue
                
                tensor_stages = []
                for stage in stage_hits:
                    stage_sorted = sorted(stage)
                    fids = torch.tensor([f[0] for f in stage_sorted], device=device)
                    scores = torch.tensor([f[1] for f in stage_sorted], device=device)
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
                    valid = (diff > MIN_FRAME_GAP) & (diff <= MAX_FRAME_GAP)
    
                    decay = torch.sigmoid((MAX_FRAME_GAP / 2 - diff.float()) / 50)
                    temp_score = dp_scores[i - 1][None, :] + curr_scores[:, None] * decay
                    temp_score = torch.where(
                        valid, temp_score, torch.full_like(temp_score, -1e9)
                    )
    
                    max_vals, max_idxs = temp_score.max(dim=1)
                    dp_scores[i] = max_vals
                    dp_paths[i] = [
                        dp_paths[i - 1][j.item()] + [k]
                        for j, k in zip(max_idxs, range(len(curr_fids)))
                    ]
    
                exist_chain = set()
                for idx, score in enumerate(dp_scores[-1]):
                    if score.item() < 0:
                        continue
                    
                    flag = True
                    frames_in_chain = set()
                    for stage_i, path in enumerate(dp_paths[-1][idx]):
                        frame_id, frame_score = tensor_stages[stage_i][2][path]
                        if frame_id in exist_chain:
                            flag = False
                            break
                        frames_in_chain.add(frame_id)
                    if not flag:
                        continue
                    
                    exist_chain.update(frames_in_chain)
                    for stage_i, path in enumerate(dp_paths[-1][idx]):
                        frame_id, frame_score = tensor_stages[stage_i][2][path]
                        all_chains.append((score.item(), (frame_id, frame_score), vid))
    
            all_chains.sort(key=lambda x: -x[0])
            all_results = all_chains
    
            await self.redis.save_tmp_search_results_to_cache(
                redis_key=redis_key, results=all_results, ttl_seconds=TIME_CACHE_QUERIES
            )
    
            t1 = time.time()
            print("CHAIN:")
            print("  algorithm_time =", t1 - t_algo)
            print("  total_time     =", t1 - t0)
    
        start = (page - 1) * page_size
        end = start + page_size
        paged = all_results[start:end]
    
        out: List[ResultItem] = []
        for i, (score, frameinfo, video_id) in enumerate(paged):
            frame_id, _frame_score = frameinfo
            out.append(
                ResultItem(
                    id=str(i + start),
                    videoId=video_id,
                    confidence=round(score, 4),
                    timestamp=str(frame_id),
                )
            )
        return out

    # ---------- /embeddings/information ----------
    async def frame_information(self, video_id: str, frame_id: int) -> Optional[InformationOfFrame]:
        """
        Replicates /information endpoint. Requires PolarService or returns None if not configured.
        """
        es_data = await asyncio.to_thread(get_text_by_frame, video_id=video_id, frame_id=frame_id)

        objects_str = ""
        if self.polar is not None:
            pl_data = await asyncio.to_thread(self.polar.get_object_by_frame, video_id, frame_id)
            if pl_data:
                objects_str = ", ".join(
                    f"{k}={v}"
                    for k, v in pl_data.items()
                    if isinstance(v, (int, float)) and v > 0
                )

        if not es_data and not objects_str:
            return None

        return InformationOfFrame(
            ocr_text=es_data.get("ocr_text", "") if es_data else "",
            asr_text=es_data.get("asr_text", "") if es_data else "",
            objects=objects_str,
        )

    # ---------- /embeddings/history ----------
    async def history(self, user_id: str, limit: int = 10):
        return await self.redis.get_queries_history(user_id=user_id, limit=limit)

    # ---------- /web/ddg_images ----------
    async def ddg_images(
        self,
        query: str,
        *,
        max_results: int = 10,
        region: str = "wt-wt",
        safesearch: str = "moderate",
        user_id: str = "agent",
        ttl_seconds: int = 60 * 10,
    ) -> list[dict]:
        """
        DuckDuckGo image search via LangChain wrapper.
        Returns a list of dicts: {title, page_url, image_url, thumbnail_url, source}
        """
        redis_key = f"tmp:ddg_images:{user_id}:{region}:{safesearch}:{max_results}:{query}".lower()

        cached = await self.redis.redis_client.get(redis_key)
        if cached is not None:
            try:
                return pickle.loads(cached)
            except Exception:
                pass

        def _run():
            return _DDG_IMAGE_WRAPPER.results(
                query=query,
                max_results=max_results,
                source="images",
            )

        try:
            results = await asyncio.to_thread(_run)
        except Exception as e:
            raise RuntimeError(f"DuckDuckGo image search failed: {e}")

        norm = []
        for r in results[:max_results]:
            norm.append({
                "title": r.get("title") or "",
                "page_url": r.get("link") or "",
                "image_url": r.get("image") or "",
                "thumbnail_url": r.get("thumbnail") or "",
                "source": r.get("source") or "duckduckgo",
            })

        try:
            await self.redis.redis_client.setex(redis_key, ttl_seconds, pickle.dumps(norm))
        except Exception:
            pass

        return norm

    # ---------- /web/ddg_web ----------
    async def ddg_web(
        self,
        query: str,
        max_results: int = 10,
        user_id: str = "agent",
        ttl_seconds: int = 600
    ) -> list[dict]:
        """
        DuckDuckGo web search via LangChain wrapper.
        Returns a list of dicts: {title, url, snippet}
        """
        key = f"tmp:ddg_web:{user_id}:{max_results}:{query}".lower()
        cached = await self.redis.redis_client.get(key)
        if cached:
            try:
                return pickle.loads(cached)
            except Exception:
                pass

        def _run():
            return _DDG_WEB_WRAPPER.results(query=query, max_results=max_results, source="text")

        try:
            rows = await asyncio.to_thread(_run)
        except Exception as e:
            raise RuntimeError(f"DuckDuckGo web search failed: {e}")

        norm = [
            {
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": r.get("snippet", "")
            }
            for r in rows[:max_results]
        ]
        
        try:
            await self.redis.redis_client.setex(key, ttl_seconds, pickle.dumps(norm))
        except Exception:
            pass
        
        return norm