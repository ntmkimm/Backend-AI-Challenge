import os
import glob
import csv  # (kept in case you later add CSV export)
from pathlib import Path

import numpy as np
from tqdm import tqdm
from pymilvus import (
    connections, utility, Collection,
    CollectionSchema, FieldSchema, DataType
)
import torch
import torch.multiprocessing as mp

# === CONFIG ===
DIMENSION = 1024
MILVUS_HOST = "192.168.20.156"
MILVUS_PORT = "19530"
BATCH_SIZE = 1024          # batch insert to Milvus (vectors only; no GPU needed here)
FLUSH_INTERVAL = 20000    # flush every N rows

COLLECTION_NAME = 'AIC25_batch1_beit3'
NPZ_KEY = "embedding"
SAVE_FOLDER_NAME = "beit3_vector"

# COLLECTION_NAME = 'AIC25_batch1_openclip'
# NPZ_KEY = "feature"
# SAVE_FOLDER_NAME = "vector_file"

ROOT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/batch1")

def encode_worker(rank: int, world_size: int, indexed_paths, device_ids):
    """
    Worker: loads precomputed npz vectors and inserts to Milvus.
    `indexed_paths` is a list of (id, image_path) tuples.
    We shard it by worker: my_paths = indexed_paths[rank::world_size]
    """
    # Connect Milvus in this process
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(name=COLLECTION_NAME)
    collection.load()

    my_paths = indexed_paths[rank::world_size]

    # Buffers for batched insert
    buffer = {'ids': [], 'paths': [], 'npz': [], 'videos': [], 'frames': []}
    inserted_since_flush = 0

    pbar = tqdm(my_paths, desc=f"[Worker {rank}] ingest", position=rank, leave=False)
    for _id, img_path in pbar:
        try:
            img_path = Path(img_path)
            # vector_file is sibling of keyframes folder
            vec_path = img_path.parent.parent / SAVE_FOLDER_NAME / (img_path.stem + ".npz")
            if not vec_path.exists():
                # skip if vector missing
                continue

            # Load npz -> 'feature' (shape [DIMENSION])
            try:
                feat = np.load(vec_path)[NPZ_KEY]
            except Exception as e:
                print(f"[Worker {rank}] Failed to load {vec_path}: {e}")
                continue

            if feat.ndim != 1 or feat.shape[0] != DIMENSION:
                print(f"[Worker {rank}] Bad dim for {vec_path}: {feat.shape}, expected ({DIMENSION},)")
                continue

            # Fill buffers
            buffer['ids'].append(int(_id))
            buffer['paths'].append(str(img_path))
            buffer['npz'].append(feat.astype(np.float32))
            buffer['videos'].append(img_path.parent.parent.name)  # video folder name
            # keyframe_XXXX.webp -> XXXX
            try:
                frame_id = int(img_path.stem.replace("keyframe_", ""))
            except ValueError:
                # Fallback: try to parse digits anywhere in stem
                digits = ''.join(ch for ch in img_path.stem if ch.isdigit())
                frame_id = int(digits) if digits else -1
            buffer['frames'].append(frame_id)

            # If batch is ready, insert
            if len(buffer['npz']) >= BATCH_SIZE:
                try:
                    # Milvus expects column-major field lists
                    collection.insert([
                        buffer['ids'],
                        # buffer['paths'],
                        buffer['npz'],
                        buffer['videos'],
                        buffer['frames'],
                    ])
                    inserted_since_flush += len(buffer['ids'])
                except Exception as e:
                    print(f"[Worker {rank}] Insert error: {e}")
                finally:
                    buffer = {'ids': [], 'paths': [], 'npz': [], 'videos': [], 'frames': []}

                # Periodic flush
                if inserted_since_flush >= FLUSH_INTERVAL:
                    try:
                        collection.flush()
                    except Exception as e:
                        print(f"[Worker {rank}] Flush error: {e}")
                    inserted_since_flush = 0

        except Exception as e:
            print(f"[Worker {rank}] Error with {img_path}: {e}")
            try:
                with open(f"fail_load_worker{rank}.txt", "a+", encoding="utf-8") as f:
                    f.write(str(img_path) + "\n")
            except Exception as err:
                print(f"[Worker {rank}] Failed to log error: {err}")

    # Final insert for remaining buffer
    if len(buffer['npz']) > 0:
        try:
            collection.insert([
                buffer['ids'],
                # buffer['paths'],
                buffer['npz'],
                buffer['videos'],
                buffer['frames'],
            ])
        except Exception as e:
            print(f"[Worker {rank}] Final insert error: {e}")
        buffer = {'ids': [], 'paths': [], 'npz': [], 'videos': [], 'frames': []}

    # Final flush
    try:
        collection.flush()
    except Exception as e:
        print(f"[Worker {rank}] Final flush error: {e}")

    print(f"[Worker {rank}] Done.")


def main():
    # Connect control-plane client
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    # Create collection if needed (drop if exists)
    if utility.has_collection(COLLECTION_NAME):
        print(f"⚠️ Collection {COLLECTION_NAME} exists. Drop and recreate? [y/n]")
        p = input().strip().lower()
        if p != 'y':
            print("Abort.")
            return
        utility.drop_collection(COLLECTION_NAME)

    # Define schema
    schema = CollectionSchema([
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        # FieldSchema(name="filepath", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="frame_id", dtype=DataType.INT64),
    ])
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    # (Optional) create index *before* or *after* insert. For huge data, after-insert is faster.
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {
            "M": 32,
            "efConstruction": 512
        }
    }
    collection.create_index("embedding", index_params)
    collection.load()

    # Gather image paths (we trust vectors exist next to them)
    image_paths = []
    # ROOT/<video>/keyframes/*.webp
    for video_dir in sorted(ROOT.iterdir()):
        kf_dir = video_dir / "keyframes"
        if kf_dir.exists():
            image_paths.extend(sorted(kf_dir.glob("*.webp")))

    # Enumerate to generate stable IDs
    indexed_paths = list(enumerate(image_paths))

    # Spawn workers (use number of GPUs if you want, but GPU not required here)
    world_size = max(1, torch.cuda.device_count())  # can be >1 even if not using GPU compute
    device_ids = list(range(world_size))
    print(f"Workers: {world_size}, total images: {len(indexed_paths)}")

    mp.spawn(encode_worker, args=(world_size, indexed_paths, device_ids), nprocs=world_size, join=True)

    # After ingestion, you can (optionally) load+compact
    print("All workers done. Compacting segments and building index...")
    try:
        collection = Collection(name=COLLECTION_NAME)
        collection.flush()
        # Optionally release/load to enforce index
        collection.release()
        collection.load()
    except Exception as e:
        print(f"Post-ingest error: {e}")

    print("✅ Milvus ingestion complete.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
