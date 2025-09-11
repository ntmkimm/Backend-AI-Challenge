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
# CLIP_MODEL = "metaclip"

CLIP_MODEL = "openclip"

# COLLECTION MILVUS
OPENCLIP_MILVUS = "AIC25_openclip"
BEIT3_MILVUS = "AIC25_beit3"
SIGLIP2_MILVUS = "AIC25_siglip2"

DEVICE_0 = "cuda:0" 
DEVICE_1 = "cuda:1"

OBJECT_DATABASE = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge/objects.parquet"

PARAPHRASE_MODEL = "humarin/chatgpt_paraphraser_on_T5_base"

# Redis Settings
REDIS_HOST = "redis-server"
REDIS_PORT = 6379

# Search Settings
TOP_K = 1000
MIN_FRAME_GAP = 50
MAX_FRAME_GAP = 750 * 4
BATCH_SIZE = 128

TIME_CACHE_ONE_QUERY = 75
TIME_CACHE_QUERIES = 75

# Media Server
# MEDIA_SERVER_URL = "http://192.168.20.156:9000/aic2025"

# Add backend to path
BACKEND_PATH = str(Path(__file__).parent.parent.parent)
if BACKEND_PATH not in sys.path:
    sys.path.append(BACKEND_PATH) 