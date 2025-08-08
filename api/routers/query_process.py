from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List
import uuid, asyncio, json
from dependencies.services import get_paraphrase_service, get_redis_service
from services.paraphrase_service import ParaphraseService
from services.redis_service import RedisService
from core.utils import get_valid_queries
from models.schemas import Query

router = APIRouter(prefix="/query_process")

@router.post("/paraphrase/text_stage", response_model=List[str])
async def paraphrase_text_stage(
    stage_number: int,
    queries: List[Query],
    redis_service: RedisService = Depends(get_redis_service),
    paraphrase_service: ParaphraseService = Depends(get_paraphrase_service)
):
    if not 1 <= stage_number <= len(queries):
        raise HTTPException(status_code=400, detail="stage_number out of range")
    queries = get_valid_queries(queries)  
    query = queries[stage_number - 1] # # Chọn đúng stage cần search
    
    if not query.text or not query.text.strip():
        raise HTTPException(status_code=400, detail="text is empty")
    try:
        result = await redis_service.get_query_text_paraphrase_cached_redis(
            query_text=query.text,
            paraphrase_service=paraphrase_service,
            ttl_seconds=300
        )
        return result  # list[str]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
