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
COLLECTION_NAME = "AIC25_fullbatch1"

# CLIP Model Settings
CLIP_MODEL = "ViT-H-14-378-quickgelu"

DEVICE = "cuda"  # Will be overridden based on availability

OBJECT_DATABASE = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1/objects.parquet"

# Redis Settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379

# Polar Settings
POLAR_HOST = "localhost"
POLAR_PORT = 5432

# Search Settings
TOP_K = 500
MAX_FRAME_GAP = 750
BATCH_SIZE = 128

# Media Server
MEDIA_SERVER_URL = "http://192.168.20.156:9000/aic2025"

# Add backend to path
BACKEND_PATH = str(Path(__file__).parent.parent.parent)
if BACKEND_PATH not in sys.path:
    sys.path.append(BACKEND_PATH) 