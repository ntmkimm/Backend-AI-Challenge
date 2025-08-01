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

from config.settings import CORS_ORIGINS
from routers import search

app = FastAPI(
    title="Embeddings API",
    docs_url="/embeddings/docs",
    openapi_url="/embeddings/openapi.json"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Include routers
app.include_router(search.router)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}
