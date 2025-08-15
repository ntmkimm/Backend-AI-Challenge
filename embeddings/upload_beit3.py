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
from pymilvus import (
    Collection, CollectionSchema, DataType, FieldSchema,
    connections, utility
)
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
from torchscale.architecture.config import EncoderConfig
from torchscale.model.BEiT3 import BEiT3
from torchvision import transforms
from torchvision.datasets.folder import default_loader
from tqdm import tqdm

# === CONFIGURATION ===
# Milvus Configuration
COLLECTION_NAME = 'AIC25_BEiT3_fullbatch1'
DIMENSION = 1024  # BEiT3-large embedding dimension
MILVUS_HOST = "192.168.20.156"
MILVUS_PORT = "19530"

# Processing Configuration
BATCH_SIZE = 48 # Adjust based on GPU memory
FLUSH_INTERVAL = 2000 # Number of embeddings to insert before flushing
ROOT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1")

# Model Configuration
MODEL_NAME = "beit3_large_patch16_384_retrieval"
MODEL_FOLDER = Path('/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/models')
CKPT_PATH = MODEL_FOLDER / ('beit3_large_patch16_384_coco_retrieval' + '.pth')
IMAGE_INPUT_SIZE = 384


# === BEiT3 MODEL DEFINITIONS ===

def trunc_normal_(tensor, mean=0., std=1.):
    """Truncated normal initialization."""
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0,
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
):
    """Get configuration for BEiT3-Large model."""
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16,
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24,
        checkpoint_activations=checkpoint_activations,
    )

class BEiT3Wrapper(nn.Module):
    """Wrapper for the BEiT3 model."""
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class BEiT3ForRetrieval(BEiT3Wrapper):
    """BEiT3 model fine-tuned for retrieval tasks."""
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
            return vision_cls, None # Return only vision embeddings for this use case

        if text_description is not None:
            # This part is not used for image indexing but is kept for completeness
            outputs = self.beit3(
                textual_tokens=text_description,
                visual_tokens=None,
                text_padding_position=padding_mask,
            )
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)
            return None, language_cls

        # During inference, we expect either an image or text, not both for loss calculation
        if only_infer:
             return vision_cls, language_cls
        
        # The following loss calculation part is not used in this indexing script
        loss, _, _ = self.criterion(
            vision_cls, language_cls, self.logit_scale.exp())
        return loss, vision_cls, language_cls


@register_model
def beit3_large_patch16_384_retrieval(pretrained=False, **kwargs):
    """Model registration with timm."""
    args = _get_large_config(img_size=IMAGE_INPUT_SIZE, **kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model

def build_transform(input_size):
    """Builds the image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


# === MILVUS INDEXING WORKER ===

def encode_worker(rank, world_size, indexed_paths, device_ids):
    """
    The worker process that loads a model, encodes images, and inserts them into Milvus.
    """
    # 1. Setup device and model
    device = torch.device(f"cuda:{device_ids[rank]}")
    
    # Load BEiT3 model
    model = timm.create_model(MODEL_NAME, pretrained=False)
    checkpoint = torch.load(CKPT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device).eval()
    
    # Get the image preprocessing pipeline
    preprocess = build_transform(IMAGE_INPUT_SIZE)

    # 2. Connect to Milvus
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(name=COLLECTION_NAME)

    # 3. Process data
    my_paths = indexed_paths[rank::world_size]
    buffer = {'images': [], 'ids': [], 'paths': [], 'videos': [], 'frames': []}
    inserted_count = 0

    for _id, path in tqdm(my_paths, desc=f"[GPU {rank}]"):
        try:
            img = Image.open(path).convert("RGB")
            buffer['images'].append(preprocess(img))
            buffer['ids'].append(_id)
            buffer['paths'].append(str(path))
            buffer['videos'].append(path.parent.parent.name)
            buffer['frames'].append(int(path.stem.replace("keyframe_", "")))

            if len(buffer['images']) >= BATCH_SIZE:
                with torch.no_grad(), torch.amp.autocast("cuda"):
                    inputs = torch.stack(buffer['images']).to(device)
                    # Get embeddings from BEiT3 model
                    vision_embs, _ = model(image=inputs, only_infer=True)
                    embs_list = vision_embs.cpu().tolist()

                try:
                    collection.insert([
                        buffer['ids'],
                        buffer['paths'],
                        embs_list,
                        buffer['videos'],
                        buffer['frames']
                    ])
                    inserted_count += len(buffer['ids'])
                    if inserted_count >= FLUSH_INTERVAL:
                        collection.flush()
                        inserted_count = 0
                except Exception as e:
                    print(f"[GPU {rank}] Milvus insert error: {e}")
                finally:
                    # Clear buffer and release GPU memory
                    buffer = {'images': [], 'ids': [], 'paths': [], 'videos': [], 'frames': []}
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"[GPU {rank}] Error processing {path}: {e}")
            with open("failed_to_process.txt", "a") as f:
                f.write(f"{path}\n")

    # 4. Final flush for any remaining items in the buffer
    if buffer['images']:
        with torch.no_grad(), torch.amp.autocast("cuda"):
            inputs = torch.stack(buffer['images']).to(device)
            vision_embs, _ = model(image=inputs, only_infer=True)
            embs_list = vision_embs.cpu().tolist()
        try:
            collection.insert([
                buffer['ids'],
                buffer['paths'],
                embs_list,
                buffer['videos'],
                buffer['frames']
            ])
            collection.flush()
        except Exception as e:
            print(f"[GPU {rank}] Final Milvus insert error: {e}")
    
    connections.disconnect(f"worker_{rank}")
    print(f"✅ [GPU {rank}] Processing complete.")


# === MAIN EXECUTION ===

def main():
    # 1. Connect to Milvus and setup collection
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    if utility.has_collection(COLLECTION_NAME):
        print(f"⚠️ Collection '{COLLECTION_NAME}' already exists.")
        p = input("Do you want to delete it and start over? [y/n]: ")
        if p.lower() != 'y':
            print("Exiting.")
            return
        utility.drop_collection(COLLECTION_NAME)
        print("Collection dropped.")

    print(f"Creating new collection: '{COLLECTION_NAME}'")
    schema = CollectionSchema([
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="filepath", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=300),
        FieldSchema(name="frame_id", dtype=DataType.INT64),
    ])
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    
    print("Creating HNSW index...")
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 32, "efConstruction": 512}
    }
    collection.create_index("embedding", index_params)
    collection.load()

    # 2. Find all image paths
    print(f"Scanning for images in {ROOT}...")
    image_paths = []
    for sub in sorted(ROOT.iterdir()):
        if sub.is_dir():
            kf_dir = sub / "keyframes"
            if kf_dir.exists():
                image_paths.extend(sorted(kf_dir.glob("*.webp")))
    
    if not image_paths:
        print("Error: No images found. Please check the ROOT directory.")
        return
        
    print(f"Found {len(image_paths)} images to process.")
    indexed_paths = list(enumerate(image_paths))

    # 3. Start parallel processing
    device_ids = list(range(torch.cuda.device_count()))
    if not device_ids:
        print("Error: No CUDA devices found. This script requires at least one GPU.")
        return
        
    world_size = len(device_ids)
    print(f"Spawning {world_size} worker processes for GPUs: {device_ids}")
    mp.spawn(encode_worker, args=(world_size, indexed_paths, device_ids), nprocs=world_size)

    print("\n--- All workers finished ---")
    print(f"Total entities in collection: {collection.num_entities}")
    connections.disconnect("default")


if __name__ == "__main__":
    # Using 'spawn' start method is crucial for CUDA in multiprocessing
    mp.set_start_method("spawn", force=True)
    main()