from services.clip_service import CLIPService
from services.milvus_service import MilvusService
from services.redis_service import RedisService
from services.polar_service import PolarService

from functools import lru_cache

@lru_cache(maxsize=1)
def get_clip_service():
    return CLIPService()

@lru_cache(maxsize=1)
def get_milvus_service():
    return MilvusService()

@lru_cache(maxsize=1)
def get_redis_service():
    return RedisService()

@lru_cache(maxsize=1)
def get_polar_service():
    return PolarService()
