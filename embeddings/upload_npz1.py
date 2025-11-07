import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from pymilvus import (
    connections, utility, Collection,
    CollectionSchema, FieldSchema, DataType
)

# === CONFIG ===
DIMENSION = 1024
MILVUS_HOST = "192.168.20.150"
MILVUS_PORT = "6050"
BATCH_SIZE = 1024
FLUSH_INTERVAL = 20000

# COLLECTION_NAME = 'AIC25_beit3'
# NPZ_KEY = "embedding"
# SAVE_FOLDER_NAME = "beit3_vector"

COLLECTION_NAME = 'AIC25_openclip'
NPZ_KEY = "feature"
SAVE_FOLDER_NAME = "vector_file"

ROOT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge")


def insert_to_milvus(indexed_paths):
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(name=COLLECTION_NAME)
    collection.load()

    buffer = {'ids': [], 'npz': [], 'videos': [], 'frames': []}
    inserted_since_flush = 0

    for _id, img_path in tqdm(indexed_paths, desc="Ingesting"):
        try:
            img_path = Path(img_path)
            vec_path = img_path.parent.parent / SAVE_FOLDER_NAME / (img_path.stem + ".npz")
            if not vec_path.exists():
                continue

            feat = np.load(vec_path)[NPZ_KEY]
            if feat.ndim != 1 or feat.shape[0] != DIMENSION:
                print(f"❌ Bad dim for {vec_path}: {feat.shape}, expected ({DIMENSION},)")
                continue

            buffer['ids'].append(int(_id))
            buffer['npz'].append(feat.astype(np.float32))
            buffer['videos'].append(img_path.parent.parent.name)

            try:
                frame_id = int(img_path.stem.replace("keyframe_", ""))
            except ValueError:
                digits = ''.join(ch for ch in img_path.stem if ch.isdigit())
                frame_id = int(digits) if digits else -1
            buffer['frames'].append(frame_id)

            if len(buffer['npz']) >= BATCH_SIZE:
                try:
                    collection.insert([
                        buffer['ids'],
                        buffer['npz'],
                        buffer['videos'],
                        buffer['frames'],
                    ])
                    inserted_since_flush += len(buffer['ids'])
                except Exception as e:
                    print(f"⚠️ Insert error: {e}")
                finally:
                    buffer = {'ids': [], 'npz': [], 'videos': [], 'frames': []}

                if inserted_since_flush >= FLUSH_INTERVAL:
                    try:
                        collection.flush()
                    except Exception as e:
                        print(f"⚠️ Flush error: {e}")
                    inserted_since_flush = 0

        except Exception as e:
            print(f"⚠️ Error with {img_path}: {e}")
            with open("fail_load.txt", "a+", encoding="utf-8") as f:
                f.write(str(img_path) + "\n")

    if len(buffer['npz']) > 0:
        try:
            collection.insert([
                buffer['ids'],
                buffer['npz'],
                buffer['videos'],
                buffer['frames'],
            ])
        except Exception as e:
            print(f"⚠️ Final insert error: {e}")

    try:
        collection.flush()
    except Exception as e:
        print(f"⚠️ Final flush error: {e}")

    print("✅ Done inserting all vectors.")


def main():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

    if utility.has_collection(COLLECTION_NAME):
        print(f"⚠️ Collection {COLLECTION_NAME} exists. Drop and recreate? [y/n]")
        p = input().strip().lower()
        if p != 'y':
            print("Abort.")
            return
        utility.drop_collection(COLLECTION_NAME)

    schema = CollectionSchema([
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="frame_id", dtype=DataType.INT64),
    ])
    collection = Collection(name=COLLECTION_NAME, schema=schema)

    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 32, "efConstruction": 512}
    }
    collection.create_index("embedding", index_params)
    collection.load()

    image_paths = []
    print("check images...")
    for video_dir in sorted(ROOT.iterdir()):
        print(video_dir)
        kf_dir = video_dir / "keyframes"
        if kf_dir.exists():
            image_paths.extend(sorted(kf_dir.glob("*.webp")))

    indexed_paths = list(enumerate(image_paths))
    print(f"Total images: {len(indexed_paths)}")

    insert_to_milvus(indexed_paths)

    try:
        collection = Collection(name=COLLECTION_NAME)
        collection.flush()
        collection.release()
        collection.load()
    except Exception as e:
        print(f"Post-ingest error: {e}")

    print("✅ Milvus ingestion complete.")


if __name__ == "__main__":
    main()
