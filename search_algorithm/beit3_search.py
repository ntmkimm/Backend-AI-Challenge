import torch
import torch.nn as nn
import torch.nn.functional as F
from pymilvus import Collection, connections
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import XLMRobertaTokenizer
import timm
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig
from pathlib import Path
from typing import List

# === CONFIGURATION ===
# Milvus Configuration
# --- IMPORTANT: Use the collection where you stored BEiT3 embeddings ---
COLLECTION_NAME = 'AIC25_BEiT3_fullbatch1' 
MILVUS_HOST = "192.168.20.156"
MILVUS_PORT = "19530"
TOP_K = 10  # Number of results to retrieve

# BEiT3 Model Configuration
MODEL_NAME = "beit3_large_patch16_384_retrieval"
# --- IMPORTANT: Update this path to your actual model checkpoint ---
BEIT3_CHECKPOINT_PATH = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/models/beit3_large_patch16_384_coco_retrieval.pth' 
TOKENIZER_BEIT3_PATH = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/models/beit3.spm'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_INPUT_SIZE = 384
TEXT_MAX_LENGTH = 64


# === BEiT3 MODEL AND SERVICE DEFINITIONS ===
# This section is the same as the BEiT3Service you created earlier.

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def _get_large_config(img_size=IMAGE_INPUT_SIZE, **kwargs):
    return EncoderConfig(
        img_size=img_size, patch_size=16, vocab_size=64010, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=0, encoder_embed_dim=1024, encoder_attention_heads=16,
        encoder_ffn_embed_dim=int(1024 * 4), encoder_layers=24,
    )

class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
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
    def __init__(self, args, **kwargs):
        super(BEiT3ForRetrieval, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.language_head.apply(self._init_weights)
        self.vision_head.apply(self._init_weights)

    def forward(self, image=None, text_description=None, padding_mask=None):
        vision_cls, language_cls = None, None
        if image is not None:
            outputs = self.beit3(visual_tokens=image)
            x = outputs["encoder_out"]
            vision_cls = F.normalize(self.vision_head(x[:, 0, :]), dim=-1)
        if text_description is not None:
            outputs = self.beit3(textual_tokens=text_description, text_padding_position=padding_mask)
            x = outputs["encoder_out"]
            language_cls = F.normalize(self.language_head(x[:, 0, :]), dim=-1)
        return vision_cls, language_cls

@register_model
def beit3_large_patch16_384_retrieval(pretrained=False, **kwargs):
    args = _get_large_config(**kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model

class BEiT3Service:
    def __init__(self, checkpoint_path: str = BEIT3_CHECKPOINT_PATH):
        print("Initializing BEiT3 Service...")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")

        self.device = DEVICE
        self.model = timm.create_model(MODEL_NAME, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model = self.model.to(self.device).eval()
        self.tokenizer = XLMRobertaTokenizer(TOKENIZER_BEIT3_PATH)
        print("BEiT3 model and tokenizer loaded.")

    def _tokenize_text(self, text: str, max_len: int = TEXT_MAX_LENGTH):
        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) > max_len - 2:
            token_ids = token_ids[:max_len - 2]
        tokens = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]
        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        tokens_padded = tokens + [self.tokenizer.pad_token_id] * (max_len - num_tokens)
        return tokens_padded, padding_mask

    def encode_single_text(self, text: str) -> List[float]:
        token_ids, padding_mask = self._tokenize_text(text)
        token_ids_tensor = torch.tensor([token_ids]).to(self.device)
        padding_mask_tensor = torch.tensor([padding_mask]).to(self.device)
        with torch.no_grad(), torch.amp.autocast("cuda"):
            _, text_embedding = self.model(text_description=token_ids_tensor, padding_mask=padding_mask_tensor)
        return text_embedding.cpu().tolist()


# === MAIN SEARCH SCRIPT ===
def main():
    # 1. INITIALIZE BEiT3 SERVICE
    try:
        beit3_service = BEiT3Service(checkpoint_path=BEIT3_CHECKPOINT_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please update the BEIT3_CHECKPOINT_PATH variable in the script.")
        return

    # 2. CONNECT TO MILVUS
    print(f"\nConnecting to Milvus at {MILVUS_HOST}:{MILVUS_PORT}...")
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(name=COLLECTION_NAME)
    collection.load()
    print(f"Successfully connected to collection '{COLLECTION_NAME}'.")
    print(f"Number of entities in collection: {collection.num_entities}")

    # 3. SEARCH LOOP
    while True:
        print("\n" + "="*50)
        text_query = input("Enter your text query (or type 'exit' to quit): ")
        if text_query.lower() == 'exit':
            break

        # === ENCODE TEXT QUERY USING BEiT3 ===
        print("Encoding text query...")
        query_embedding = beit3_service.encode_single_text(text_query)

        # === SEARCH IN MILVUS ===
        print("Searching in Milvus...")
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "nprobe": 128  # Adjust for a balance between speed and accuracy
            }
        }
        
        results = collection.search(
            data=query_embedding,
            anns_field="embedding",  # The field name in your BEiT3 collection
            param=search_params,
            limit=TOP_K,
            output_fields=["filepath", "video_id"] # Specify fields to retrieve
        )

        # === SHOW RESULTS ===
        print("\n--- Search Results ---")
        hits = results[0]
        if not hits:
            print("No results found.")
            continue
            
        for i, hit in enumerate(hits):
            filepath = hit.entity.get('filepath')
            video_id = hit.entity.get('video_id')
            score = hit.distance
            
            print(f"Top {i+1}:")
            print(f"  - Score (Distance): {score:.4f}")
            print(f"  - Video ID: {video_id}")
            print(f"  - File Path: {filepath}")

    connections.disconnect("default")
    print("\nDisconnected from Milvus. Goodbye!")


if __name__ == "__main__":
    main()