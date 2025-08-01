import torch
import os
import random
from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from typing import List, Optional, Dict
from collections import defaultdict
from pydantic import BaseModel
import torch.nn.functional as F
import math

from pymilvus import Collection, connections
import open_clip
import time
import heapq
import sys
from pathlib import Path
from difflib import SequenceMatcher

sys.path.append(str(Path(__file__).parent.parent))
from utils.es_module import search_by_ocr

TOP_K = 1000
MAX_FRAME_GAP = 750
BATCH_SIZE = 128  # Optimized for RTX 2080 Ti 11GB VRAM // OOM errors thì giảm xuống 64

app = FastAPI(
    title="Embeddings API",
    docs_url="/embeddings/docs",
    openapi_url="/embeddings/openapi.json"
)

# Configure CORS
origins = [
    "http://localhost:5731",
    "http://192.168.20.156:5731",
    "http://localhost:8081"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Connect to Milvus
try:
    connections.connect(host="192.168.20.156", port="19530")
    collection = Collection("AIC25_fullbatch1")
    collection.load()
except Exception as e:
    print(f"Error connecting to Milvus: {str(e)}")
    raise

# Load CLIP model
try:
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14-378-quickgelu", pretrained="dfn5b")
    tokenizer = open_clip.get_tokenizer("ViT-H-14-378-quickgelu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
except Exception as e:
    print(f"Error loading CLIP model: {str(e)}")
    raise

class Query(BaseModel):
    Queries: List[str]

class ResultItem(BaseModel):
    id: str
    videoId: str
    title: str
    thumbnail: str
    confidence: float
    timestamp: str

class OCRSearchRequest(BaseModel):
    query: str
    size: int = 10

class OCRSearchResponse(BaseModel):
    results: List[str]

# === Simple in-memory fuzzy cache ===
cached_queries: List[Dict] = []

def fuzzy_equal(q1: List[str], q2: List[str], threshold: float = 0.95) -> bool:
    if len(q1) != len(q2):
        return False
    return all(SequenceMatcher(None, a, b).ratio() >= threshold for a, b in zip(q1, q2))

def get_cached_all_answers(current_queries: List[str]):
    for item in cached_queries:
        if fuzzy_equal(item["queries"], current_queries):
            return item["all_answers"]
    return None

def store_cache_all_answers(queries: List[str], all_answers):
    cached_queries.append({
        "queries": queries,
        "all_answers": all_answers,
        "timestamp": time.time()
    })
    MAX_CACHE_SIZE = 10
    if len(cached_queries) > MAX_CACHE_SIZE:
        del cached_queries[:len(cached_queries) - MAX_CACHE_SIZE]

# Encode text in batch

def encode_text_batch(model, tokenizer, texts, device, batch_size=BATCH_SIZE):
    num_texts = len(texts)
    embeddings_list = []
    for i in range(0, num_texts, batch_size):
        batch_texts = texts[i:i + batch_size]
        with torch.no_grad():
            tokens = tokenizer(batch_texts).to(device)
            batch_embeddings = model.encode_text(tokens)
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)
            embeddings_list.append(batch_embeddings)
    embeddings = torch.cat(embeddings_list, dim=0)
    return embeddings

import requests

def translate(text, source_lang='auto', target_lang='en'):
    url = "https://translate.googleapis.com/translate_a/single"
    params = {
        'client': 'gtx',
        'sl': source_lang,
        'tl': target_lang,
        'dt': 't',
        'q': text
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        result = response.json()
        translated_text = ''.join([item[0] for item in result[0]])
        detected_source = result[2]  # Detected source language
        return translated_text
    else:
        raise Exception(f"Translation failed with status code {response.status_code}")


@lru_cache(maxsize=128)
def cached_text_embedding(query: str):
    with torch.no_grad():
        tokens = tokenizer([query]).to(device)
        embedding = model.encode_text(tokens)
        return F.normalize(embedding, p=2, dim=-1).cpu().tolist()

@app.post("/embeddings/search", response_model=List[ResultItem])
async def search_text(query: Query):
    try:
        if not query.Queries or all(s.strip() == "" for s in query.Queries):
            raise HTTPException(status_code=400, detail="Search text cannot be empty")
        query.Queries = [translate(q) for q in query.Queries]
        cached_result = get_cached_all_answers(query.Queries)
        if cached_result is not None:
            all_answers = cached_result
        else:
            embeddings = encode_text_batch(model, tokenizer, query.Queries, device)
            all_answers = [[] for _ in range(len(embeddings))]
            for query_idx, embedding in enumerate(embeddings.cpu().tolist()):
                iterator = collection.search_iterator(
                    data=[embedding], 
                    anns_field="clip_embedding",
                    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                    limit=TOP_K,
                    batch_size=200,
                    output_fields=["filepath", "frame_id", "video_id"],
                )
                while True:
                    hits = iterator.next()
                    if not hits:
                        iterator.close()
                        break
                    all_answers[query_idx].extend(hits)
            store_cache_all_answers(query.Queries, all_answers)


        # === GROUP BY VIDEO ID ===
        video_groups = defaultdict(lambda: [[] for _ in range(len(query.Queries))])
        for stage_idx, hits in enumerate(all_answers):
            for h in hits:
                video_groups[h.entity["video_id"]][stage_idx].append((int(h.entity["frame_id"]), h.distance, h.entity["filepath"]))

        # === TEMPORAL SCORING (NO CHAIN) ===
        final_results = []

        for vid, stage_hits in video_groups.items():
            if any(len(stage) == 0 for stage in stage_hits):
                continue

            # Convert each stage to tensors
            tensor_stages = []
            for stage in stage_hits:
                stage_sorted = sorted(stage)
                fids = torch.tensor([x[0] for x in stage_sorted], device=device)
                scores = torch.tensor([x[1] for x in stage_sorted], device=device)
                tensor_stages.append((fids, scores, stage_sorted))


            base_fids, base_scores, base_raw = tensor_stages[0]
            final_scores = base_scores.clone()

            for i in range(1, len(tensor_stages)):

                curr_fids, curr_scores, _ = tensor_stages[i]
                diff = curr_fids[:, None] - base_fids[None, :]
                valid = (diff > 0) & (diff <= MAX_FRAME_GAP // len(query.Queries))

                # decay = (MAX_FRAME_GAP - diff.float()) / MAX_FRAME_GAP
                decay = torch.sigmoid((MAX_FRAME_GAP / 2 - diff.float()) / 30)
                boost = curr_scores[:, None] * decay
                boost = torch.where(valid, boost, torch.zeros_like(boost))

                num_valid = valid.sum(dim=0).clamp(min=1)
                final_scores += boost.sum(dim=0) / num_valid

            for i in range(len(base_fids)):
                frame_id, dist, path = base_raw[i]
                final_results.append((final_scores[i].item(), path, frame_id, vid))

        # === SORT AND DISPLAY RESULTS ===
        final_results.sort(key=lambda x: -x[0])
        formatted_results = []

        for i, (score, path, frame_id, video_id) in enumerate(final_results[:]):

            result_item = ResultItem(
                id=str(i),
                videoId=video_id,
                title=video_id + "/" + str(frame_id),
                thumbnail=f"http://192.168.20.156:9000/aic2025/{path}",  # Adjust this URL to your media server
                confidence=round(score, 4),
                timestamp=str(frame_id)
            )
            formatted_results.append(result_item)

        return formatted_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings/chain_search", response_model=List[ResultItem])
async def chain_search_text(query: Query):
    try:
        if not query.Queries or all(s.strip() == "" for s in query.Queries):
            raise HTTPException(status_code=400, detail="Search text cannot be empty")
        query.Queries = [translate(q) for q in query.Queries]
        cached_result = get_cached_all_answers(query.Queries)
        if cached_result is not None:
            all_answers = cached_result
        else:
            embeddings = encode_text_batch(model, tokenizer, query.Queries, device)
            all_answers = [[] for _ in range(len(embeddings))]
            for query_idx, embedding in enumerate(embeddings.cpu().tolist()):
                iterator = collection.search_iterator(
                    data=[embedding], 
                    anns_field="clip_embedding",
                    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                    limit=TOP_K,
                    batch_size=200,
                    output_fields=["filepath", "frame_id", "video_id"],
                )
                while True:
                    hits = iterator.next()
                    if not hits:
                        iterator.close()
                        break
                    all_answers[query_idx].extend(hits)
            store_cache_all_answers(query.Queries, all_answers)


        # === GROUP BY VIDEO ID ===
        video_groups = defaultdict(lambda: [[] for _ in range(len(query.Queries))])
        for stage_idx, hits in enumerate(all_answers):
            for h in hits:
                video_groups[h.entity["video_id"]][stage_idx].append((int(h.entity["frame_id"]), h.distance, h.entity["filepath"]))

        # === ALIGN AND SCORE CHAINS ===
        best_chain = None
        best_score = -1e9
        TOP_K_CHAINS = 3
        all_chains = []

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
                valid = (diff > 0) & (diff <= MAX_FRAME_GAP // len(query.Queries))

                # decay = (MAX_FRAME_GAP - diff) / MAX_FRAME_GAP / len(query.Queries)
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

        # === SHOW RESULTS ===
        formatted_results = []
        for rank, (score, (frame_id, distance_score, path), vid) in enumerate(all_chains[:TOP_K]):
            weighted_score = 0.2 * (len(all_chains) - rank) / len(all_chains) +  0.6 * score * (len(all_chains) - rank) / len(all_chains) + 0.2 * distance_score
            stage = (rank) % len(query.Queries)
            result_item = ResultItem(
                id=str(rank),
                videoId=vid,
                title=vid + "/" + str(frame_id) + "-" + str(stage),
                thumbnail=f"http://192.168.20.156:9000/aic2025/{path}",
                confidence=round(score, 4),
                timestamp=str(frame_id)
            )
            formatted_results.append(result_item)

        return formatted_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/embeddings/ocr_search", response_model=List[ResultItem])
# async def search_ocr(request: OCRSearchRequest):
#     try:
#         if not request.query or request.query.strip() == "":
#             raise HTTPException(status_code=400, detail="Search text cannot be empty")
            
#         results = search_by_ocr(request.query, request.size)
        
#         formatted_results = []
#         for i, (video_id, frame_id, score, path) in enumerate(results):
#             result_item = ResultItem(
#                 id=str(i),
#                 videoId=video_id,
#                 title=f"{video_id}/{frame_id}",
#                 thumbnail=f"http://192.168.20.156:9000/aic2025/{path}",
#                 confidence=round(score, 4),
#                 timestamp=str(frame_id)
#             )
#             formatted_results.append(result_item)
            
#         return formatted_results
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

