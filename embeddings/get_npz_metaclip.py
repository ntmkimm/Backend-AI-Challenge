import math
import os
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
from torchscale.architecture.config import EncoderConfig
from torchvision import transforms
from tqdm import tqdm
from services.MetaCLIP.src.mini_clip.factory import create_model_and_transforms, get_tokenizer

# === CONFIGURATION ===
ROOT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge")
BATCH_SIZE = 40
SAVE_FOLDER_NAME = "metaclip_vector"

def encode_worker(rank, world_size, indexed_paths, device_ids):
    device = torch.device(f"cuda:{device_ids[rank]}")

    model, _, preprocess = create_model_and_transforms('ViT-H-14-quickgelu-worldwide@WorldWideCLIP', pretrained='metaclip2_worldwide')
    model = model.to(device).eval()

    my_paths = indexed_paths[rank::world_size]

    buffer = {'images': [], 'paths': [], 'videos': [], 'frames': []}

    for _idx, path in tqdm(my_paths, desc=f"[GPU {rank}]"):
        try:
            img = Image.open(path).convert("RGB")
            buffer['images'].append(preprocess(img))
            buffer['paths'].append(str(path))
            buffer['videos'].append(path.parent.parent.name)
            buffer['frames'].append(int(path.stem.replace("keyframe_", "")))

            if len(buffer['images']) >= BATCH_SIZE:
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    inputs = torch.stack(buffer['images']).to(device)
                    vision_embs = model.encode_image(inputs).cpu()

                for i, emb in enumerate(vision_embs):
                    video_id = buffer['videos'][i]
                    save_dir = ROOT / video_id / SAVE_FOLDER_NAME
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = save_dir / f"{Path(buffer['paths'][i]).stem}.npz"
                    np.savez(save_path, embedding=emb.numpy())

                buffer = {'images': [], 'paths': [], 'videos': [], 'frames': []}
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[GPU {rank}] Error processing {path}: {e}")
            with open("failed_to_process.txt", "a") as f:
                f.write(f"{path}\n")

    if buffer['images']:
        with torch.no_grad(), torch.amp.autocast("cuda"):
            inputs = torch.stack(buffer['images']).to(device)
            vision_embs = model.encode_image(inputs).cpu()

        for i, emb in enumerate(vision_embs):
            video_id = buffer['videos'][i]
            save_dir = ROOT / video_id / SAVE_FOLDER_NAME
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{Path(buffer['paths'][i]).stem}.npz"
            np.savez(save_path, embedding=emb.numpy())

    print(f"✅ [GPU {rank}] Processing complete.")

# === MAIN FUNCTION ===

def is_valid_npz(path):
    try:
        with np.load(path, allow_pickle=False) as data:
            # check if "embedding" exists
            if "embedding" not in data:
                return False
            # optional: check shape
            arr = data["embedding"]
            if arr.ndim == 0:  # probably broken
                return False
        return True
    except Exception as e:
        print(f"Corrupted {path}: {e}")
        return False

def main():
    print(f"Scanning for images in {ROOT}...")
    image_paths = []
    for sub in tqdm(sorted(ROOT.iterdir())):
        if not (sub.name >= 'K01_V001' and sub.name < 'K21_V001'): 
            continue
        if sub.is_dir():
            kf_dir = sub / "keyframes"
            vec_dir = sub / SAVE_FOLDER_NAME
            if kf_dir.exists():
                for _img in sorted(kf_dir.glob("*.webp")):
                    npz_path = vec_dir / (_img.stem + ".npz")
                    if not npz_path.exists() or not is_valid_npz(npz_path):
                        image_paths.append(_img)

    if not image_paths:
        print("Error: No images found. Please check the ROOT directory.")
        return

    print(f"Found {len(image_paths)} images to process.")
    indexed_paths = list(enumerate(image_paths))
    print("forward")
    # indexed_paths = indexed_paths[::-1]
    # print("reverse")

    device_ids = list(range(torch.cuda.device_count()))
    if not device_ids:
        print("Error: No CUDA devices found. This script requires at least one GPU.")
        return

    world_size = len(device_ids)
    print(f"Spawning {world_size} worker processes for GPUs: {device_ids}")
    mp.spawn(encode_worker, args=(world_size, indexed_paths, device_ids), nprocs=world_size)

    print("\n--- All workers finished ---")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
