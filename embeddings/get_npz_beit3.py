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
from torchscale.model.BEiT3 import BEiT3
from torchvision import transforms
from tqdm import tqdm

# === CONFIGURATION ===
ROOT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/batch1")
BATCH_SIZE = 36
MODEL_NAME = "beit3_large_patch16_384_retrieval"
MODEL_FOLDER = Path('/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/models')
CKPT_PATH = MODEL_FOLDER / ('beit3_large_patch16_384_coco_retrieval' + '.pth')
IMAGE_INPUT_SIZE = 384
SAVE_FOLDER_NAME = "beit3_vector"

# === BEiT3 MODEL DEFINITIONS ===

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def _get_large_config(img_size=224, patch_size=16, drop_path_rate=0,
                      checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16,
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24,
        checkpoint_activations=checkpoint_activations,
    )

class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class BEiT3ForRetrieval(BEiT3Wrapper):
    def __init__(self, args, **kwargs):
        super(BEiT3ForRetrieval, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.language_head.apply(self._init_weights)
        self.vision_head.apply(self._init_weights)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image=None, text_description=None, padding_mask=None, only_infer=False, **kwargs):
        if image is not None:
            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=image,
                text_padding_position=None,
            )
            x = outputs["encoder_out"]
            vision_cls = self.vision_head(x[:, 0, :])
            vision_cls = F.normalize(vision_cls, dim=-1)
            return vision_cls, None
        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description,
                visual_tokens=None,
                text_padding_position=padding_mask,
            )
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)
            return None, language_cls
        if only_infer:
            return vision_cls, language_cls

@register_model
def beit3_large_patch16_384_retrieval(pretrained=False, **kwargs):
    args = _get_large_config(img_size=IMAGE_INPUT_SIZE, **kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model

def build_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])

# === WORKER FUNCTION ===

def encode_worker(rank, world_size, indexed_paths, device_ids):
    device = torch.device(f"cuda:{device_ids[rank]}")

    model = timm.create_model(MODEL_NAME, pretrained=False)
    checkpoint = torch.load(CKPT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device).eval()

    preprocess = build_transform(IMAGE_INPUT_SIZE)
    my_paths = indexed_paths[rank::world_size]

    buffer = {'images': [], 'paths': [], 'videos': [], 'frames': []}

    for _, path in tqdm(my_paths, desc=f"[GPU {rank}]"):
        try:
            img = Image.open(path).convert("RGB")
            buffer['images'].append(preprocess(img))
            buffer['paths'].append(str(path))
            buffer['videos'].append(path.parent.parent.name)
            buffer['frames'].append(int(path.stem.replace("keyframe_", "")))

            if len(buffer['images']) >= BATCH_SIZE:
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    inputs = torch.stack(buffer['images']).to(device)
                    vision_embs, _ = model(image=inputs, only_infer=True)
                    vision_embs = vision_embs.cpu()

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
            vision_embs, _ = model(image=inputs, only_infer=True)
            vision_embs = vision_embs.cpu()

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
    # print("forward")
    indexed_paths = indexed_paths[::-1]
    print("reverse")

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
