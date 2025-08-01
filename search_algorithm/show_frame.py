# from pathlib import Path

# ROOT = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1"

# VIDEO_ID = "L04_V019"

# FRAME_ID = "17984"

# print(ROOT + "/" + VIDEO_ID + "/keyframes/keyframe_"  + FRAME_ID + ".webp")


import redis
import json

# Setup Redis client (adjust host/port if needed)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
print(1)
