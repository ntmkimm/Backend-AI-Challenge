from pathlib import Path
import sys

# CORS Settings
CORS_ORIGINS = [
    "http://localhost:5731",
    "http://192.168.20.156:5731",
    "http://localhost:8081"
]

# Milvus Settings
MILVUS_HOST = "192.168.20.156"
MILVUS_PORT = "19530"

# NOTE: CHÚ Ý EMBEDDING
COLLECTION_NAME = "AIC25_fullbatch1_metaclip"
CLIP_MODEL = "metaclip"
ANNS_FIELD_MILVUS = 'metaclip_embedding'

# COLLECTION_NAME = "AIC25_fullbatch1"
# CLIP_MODEL = "openclip"
# ANNS_FIELD_MILVUS = 'clip_embedding'

DEVICE = "cuda"  # Will be overridden based on availability

OBJECT_DATABASE = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1/objects.parquet"

# Redis Settings
REDIS_HOST = "redis-server"
REDIS_PORT = 6379

# Search Settings
TOP_K = 500
MAX_FRAME_GAP = 750
BATCH_SIZE = 128

TIME_CACHE_ONE_QUERY = 3600
TIME_CACHE_QUERIES = 90

# Media Server
MEDIA_SERVER_URL = "http://192.168.20.156:9000/aic2025"

# Add backend to path
BACKEND_PATH = str(Path(__file__).parent.parent.parent)
if BACKEND_PATH not in sys.path:
    sys.path.append(BACKEND_PATH) 