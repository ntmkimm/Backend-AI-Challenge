import asyncio
import math
from pathlib import Path
from typing import List

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
from torchscale.architecture.config import EncoderConfig
from torchscale.model.BEiT3 import BEiT3
from torchvision import transforms
from transformers import XLMRobertaTokenizer

# === CONFIGURATION ===
# You can move these to a separate config file (e.g., config/settings.py)
MODEL_NAME = "beit3_large_patch16_384_retrieval"
# --- IMPORTANT ---
# Update this path to point to your actual BEiT3 model checkpoint file
BEIT3_CHECKPOINT_PATH = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/models/beit3_large_patch16_384_coco_retrieval.pth' 
TOKENIZER_BEIT3_PATH = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/models/beit3.spm'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # Default batch size for encoding
IMAGE_INPUT_SIZE = 384
TEXT_MAX_LENGTH = 64

# === BEiT3 MODEL DEFINITIONS ===
# This section contains the necessary model architecture code for BEiT3.

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

    def forward(self, image=None, text_description=None, padding_mask=None, only_infer=False):
        vision_cls, language_cls = None, None

        if image is not None:
            outputs = self.beit3(visual_tokens=image, text_padding_position=None)
            x = outputs["encoder_out"]
            vision_cls = self.vision_head(x[:, 0, :])
            vision_cls = F.normalize(vision_cls, dim=-1)

        if text_description is not None:
            outputs = self.beit3(textual_tokens=text_description, text_padding_position=padding_mask)
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)
        
        return vision_cls, language_cls

@register_model
def beit3_large_patch16_384_retrieval(pretrained=False, **kwargs):
    """Model registration with timm."""
    args = _get_large_config(img_size=IMAGE_INPUT_SIZE, **kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model

# === SERVICE CLASS ===

class BEiT3Service:
    def __init__(self, device, checkpoint_path: str = BEIT3_CHECKPOINT_PATH):
        print(f"Initializing BEiT3 Service...")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}. Please update the BEIT3_CHECKPOINT_PATH.")

        self.device = device
        
        # 1. Load Model
        self.model = timm.create_model(MODEL_NAME, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model = self.model.to(self.device).eval()
        print(f"BEiT3 model loaded onto {self.device}.")

        # 2. Load Tokenizer
        self.tokenizer = XLMRobertaTokenizer(TOKENIZER_BEIT3_PATH)

        # 3. Define Image Preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((IMAGE_INPUT_SIZE, IMAGE_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])

    def _tokenize_text(self, text: str, max_len: int = TEXT_MAX_LENGTH):
        """Helper function to tokenize and pad a single text."""
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
        """Encodes a single text string into a BEiT3 embedding."""
        token_ids, padding_mask = self._tokenize_text(text)
        
        token_ids_tensor = torch.tensor([token_ids]).to(self.device)
        padding_mask_tensor = torch.tensor([padding_mask]).to(self.device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            _, text_embedding = self.model(text_description=token_ids_tensor, padding_mask=padding_mask_tensor)
        
        return text_embedding.squeeze(0).cpu().tolist()

    def encode_text_batch(self, texts: List[str], batch_size: int = BATCH_SIZE) -> List[List[float]]:
        """Encodes a batch of texts into BEiT3 embeddings."""
        if not texts:
            return []

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            token_ids_list = []
            padding_masks_list = []

            for text in batch_texts:
                token_ids, padding_mask = self._tokenize_text(text)
                token_ids_list.append(token_ids)
                padding_masks_list.append(padding_mask)

            token_ids_tensor = torch.tensor(token_ids_list).to(self.device)
            padding_mask_tensor = torch.tensor(padding_masks_list).to(self.device)

            with torch.no_grad(), torch.amp.autocast("cuda"):
                _, batch_embeddings = self.model(text_description=token_ids_tensor, padding_mask=padding_mask_tensor)
            
            all_embeddings.extend(batch_embeddings.cpu().tolist())

        return all_embeddings

    def encode_image(self, image: Image.Image) -> List[float]:
        """Encodes a single PIL Image (RGB) into a BEiT3 embedding."""
        if not isinstance(image, Image.Image):
            raise TypeError(f"Input must be a PIL.Image.Image, got {type(image)}")
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            image_embedding, _ = self.model(image=img_tensor)
        
        return image_embedding.squeeze(0).cpu().tolist()

    def encode_image_batch(self, images: List[Image.Image]) -> np.ndarray:
        """Encodes a batch of PIL Images into BEiT3 embeddings."""
        if not images:
            return np.empty((0, self.model.beit3.encoder.embed_dim))

        processed_tensors = []
        for img in images:
            if not isinstance(img, Image.Image):
                raise TypeError(f"Input must be a PIL.Image.Image, got {type(img)}")
            if img.mode != "RGB":
                img = img.convert("RGB")
            processed_tensors.append(self.preprocess(img))
        
        batch_tensor = torch.stack(processed_tensors, dim=0).to(self.device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            features, _ = self.model(image=batch_tensor)
        
        return features.cpu().numpy()

    async def encode_text_batch_async(self, texts: List[str], batch_size: int = BATCH_SIZE) -> List[List[float]]:
        """Asynchronously encodes a batch of texts."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.encode_text_batch(texts, batch_size))

    async def encode_image_async(self, image: Image.Image) -> List[float]:
        """Asynchronously encodes a single image."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.encode_image(image))

