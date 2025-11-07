from pathlib import Path
import sys
from dotenv import load_dotenv
import os

# Load variables from .env file into environment
load_dotenv()

# Now you can access them
tracing = os.getenv("LANGSMITH_TRACING")
endpoint = os.getenv("LANGSMITH_ENDPOINT")
api_key = os.getenv("LANGSMITH_API_KEY")
project = os.getenv("LANGSMITH_PROJECT")


# CORS Settings
CORS_ORIGINS = [
    "http://localhost:5732",
    # "http://192.168.20.150:5731",
    "http://192.168.20.150:5732",
    "http://localhost:8081",
    "http://localhost:8091"
]

# Milvus Settings
MILVUS_HOST = "192.168.20.150"
MILVUS_PORT = "19532"

# NOTE: CHÚ Ý EMBEDDING
# CLIP_MODEL = "metaclip"

CLIP_MODEL = "openclip"

# COLLECTION MILVUS
OPENCLIP_MILVUS = "quannh_AIC25_openclip"
BEIT3_MILVUS = "quannh_AIC25_beit3"
SIGLIP2_MILVUS = "quannh_AIC25_siglip2"

DEVICE_0 = "cuda:0" 
DEVICE_1 = "cuda:1"

OBJECT_DATABASE = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge/objects.parquet"

INTERVAL_JSON_FILE = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/interval.json"

# Redis Settings
REDIS_HOST = "redis-server"
REDIS_PORT = 6379

# Search Settings
TOP_K = 5000
MIN_FRAME_GAP = 24
MAX_FRAME_GAP = 25 * 60 * 20 # 20 phút
BATCH_SIZE = 128

TIME_CACHE_ONE_QUERY = 75
TIME_CACHE_QUERIES = 75

# Media Server
# MEDIA_SERVER_URL = "http://192.168.20.156:9000/aic2025"

# Add backend to path
BACKEND_PATH = str(Path(__file__).parent.parent.parent)
if BACKEND_PATH not in sys.path:
    sys.path.append(BACKEND_PATH) 