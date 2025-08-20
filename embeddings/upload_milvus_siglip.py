import os
import torch
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from pymilvus import (
    connections, utility, Collection,
    CollectionSchema, FieldSchema, DataType, exceptions
)
import torch.multiprocessing as mp
from transformers import AutoProcessor, AutoModel

# === CONFIG ===
COLLECTION_NAME = 'AIC25_fullbatch1_siglip2'
DIMENSION = 1536
MILVUS_HOST = "192.168.20.156"
MILVUS_PORT = "19530"
BATCH_SIZE = 16
FLUSH_INTERVAL = 2000
ROOT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1")


def encode_worker(rank, world_size, indexed_paths, device_ids):
    torch.cuda.set_device(device_ids[rank])
    device = torch.device(f"cuda:{device_ids[rank]}")

    ckpt = "google/siglip2-giant-opt-patch16-256"
    model = AutoModel.from_pretrained(ckpt, attn_implementation="sdpa").to(device).eval()
    processor = AutoProcessor.from_pretrained(ckpt)

    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(name=COLLECTION_NAME)
    collection.load()

    my_paths = indexed_paths[rank::world_size]
    buffer = {'images': [], 'ids': [], 'paths': [], 'videos': [], 'frames': []}
    inserted_count = 0

    for _id, path in tqdm(my_paths, desc=f"[GPU {rank}]"):
        try:
            img = Image.open(path).convert("RGB")
            buffer['images'].append(img)
            buffer['ids'].append(_id)
            buffer['paths'].append(str(path))
            buffer['videos'].append(path.parent.parent.name)
            buffer['frames'].append(int(path.stem.replace("keyframe_", "")))

            if len(buffer['images']) >= BATCH_SIZE:
                with torch.no_grad():
                    inputs = processor(images=buffer['images'], return_tensors="pt").to(device)
                    embs = model.get_image_features(**inputs)  # shape (B, 1536)
                    embs = embs.cpu().tolist()

                try:
                    collection.insert([
                        buffer['ids'],
                        buffer['paths'],
                        embs,
                        buffer['videos'],
                        buffer['frames']
                    ])
                    inserted_count += len(buffer['ids'])
                    if inserted_count >= FLUSH_INTERVAL:
                        collection.flush()
                        inserted_count = 0
                except Exception as e:
                    print(f"[GPU {rank}] Insert error: {e}")
                finally:
                    buffer = {'images': [], 'ids': [], 'paths': [], 'videos': [], 'frames': []}
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"[GPU {rank}] Error with {path}: {e}")
            try:
                with open("fail_load.txt", "a+") as f:
                    f.write(path + "\n")
                # os.remove(path)
            except Exception as err:
                print(f"[GPU {rank}] Failed to delete: {err}")

    # Final flush
    if buffer['images']:
        with torch.no_grad():
            inputs = processor(images=buffer['images'], return_tensors="pt").to(device)
            embs = model.get_image_features(**inputs).cpu().tolist()
        try:
            collection.insert([
                buffer['ids'],
                buffer['paths'],
                embs,
                buffer['videos'],
                buffer['frames']
            ])
        except Exception as e:
            print(f"[GPU {rank}] Final insert error: {e}")
        finally:
            collection.flush()
            torch.cuda.empty_cache()

    print(f"[GPU {rank}] Done.")

def main():
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(COLLECTION_NAME):
        print(f"⚠️ Collection {COLLECTION_NAME} exists. Do you want to delete it [y/n]")
        p = input()
        if (p != 'y'): return 
        utility.drop_collection(COLLECTION_NAME)

    schema = CollectionSchema([
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="filepath", dtype=DataType.VARCHAR, max_length=300),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=300),
        FieldSchema(name="frame_id", dtype=DataType.INT64),
    ])
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create_index("embedding", {
        "metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 16384}
    })
    collection.load()

    image_paths = []
    for sub in sorted(ROOT.iterdir()):
        kf_dir = sub / "keyframes"
        if kf_dir.exists():
            image_paths += sorted(kf_dir.glob("*.webp"))

    indexed_paths = list(enumerate(image_paths))
    device_ids = list(range(torch.cuda.device_count()))

    mp.spawn(encode_worker, args=(len(device_ids), indexed_paths, device_ids), nprocs=len(device_ids))

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
