import os
import psycopg2
import re
from tqdm import tqdm
from pathlib import Path
import numpy as np

# DB config
DB_NAME = "keyframes_db"
DB_USER = "quannh"
DB_PASSWORD = "1"
DB_HOST = "192.168.20.156"
DB_PORT = 5432

# Dataset paths
BASE_PATH = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/batch1")
LABEL_PATH = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/cluster/cluster_labels_100.npy")

# Load keyframes
keyframe_files = []
for _video in tqdm(sorted(BASE_PATH.iterdir())):
    path = _video / "keyframes"
    _keyframes = sorted(path.glob("*.webp"))
    keyframe_files.extend(_keyframes)

# Load labels
labels = np.load(LABEL_PATH)
assert len(labels) == len(keyframe_files), f"Mismatch: {len(labels)} labels vs {len(keyframe_files)} keyframes"

# Insert into DB
inserted = 0

try:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

    with conn:
        with conn.cursor() as cur:
            for _id, _keyframe in tqdm(enumerate(keyframe_files), desc="Inserting into DB", total=len(keyframe_files)):
                _label = int(labels[_id])
                _video_id = _keyframe.parent.parent.stem
                _frame_id =int(_keyframe.stem[9:])

                try:
                    cur.execute("""
                        INSERT INTO cluster (video_id, frame_id, label)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (video_id, frame_id) DO NOTHING;
                    """, (_video_id, _frame_id, _label))
                    inserted += 1
                except Exception as e:
                    print(f"Error inserting {str(_keyframe)}: {e}")

except KeyboardInterrupt:
    print("\nInterrupted by user. Exiting gracefully.")
finally:
    if 'conn' in locals() and conn:
        conn.close()

print("Total keyframes inserted:", inserted)
