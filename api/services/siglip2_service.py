import torch
import torch.nn.functional as F
import numpy as np
from typing import List
from transformers import AutoProcessor, AutoModel
from PIL import Image

class SigLIP2Service:
    def __init__(self, ckpt="google/siglip2-giant-opt-patch16-384", device="cuda"):
        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Load processor + model
        print("Load siglip2")
        self.processor = AutoProcessor.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(
            ckpt,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()

        # SigLIP2 docs: max_length = 64 cho text
        self.max_length = 64

    # -------------------------------
    # TEXT
    # -------------------------------
    def encode_text_batch(self, texts: List[str], batch_size: int = 16):
        num_texts = len(texts)
        embeddings_list = []

        for i in range(0, num_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.processor(
                text=batch_texts,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)

            text_inputs = {k: v for k, v in inputs.items()
                          if k in ["input_ids", "attention_mask"]}

            with torch.no_grad():
                batch_emb = self.model.get_text_features(**text_inputs)
                batch_emb = F.normalize(batch_emb, p=2, dim=-1)  # giống CLIPService
                embeddings_list.append(batch_emb)

        embeddings = torch.cat(embeddings_list, dim=0)
        return embeddings.tolist()

    def encode_single_text(self, text: str):
        inputs = self.processor(
            text=[text],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        text_inputs = {k: v for k, v in inputs.items()
                      if k in ["input_ids", "attention_mask"]}

        with torch.no_grad():
            emb = self.model.get_text_features(**text_inputs)
            emb = F.normalize(emb, p=2, dim=-1).squeeze(0)

        return emb.cpu().tolist()

    # -------------------------------
    # IMAGE
    # -------------------------------
    def encode_image_batch(self, images: List[Image.Image], batch_size: int = 4):
        embeddings_list = []

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]
            # convert về RGB
            batch_imgs = [img.convert("RGB") if img.mode != "RGB" else img for img in batch_imgs]

            inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.device)

            with torch.no_grad():
                batch_emb = self.model.get_image_features(**inputs)
                batch_emb = F.normalize(batch_emb, p=2, dim=-1)
                embeddings_list.append(batch_emb)

        embeddings = torch.cat(embeddings_list, dim=0)
        return embeddings

    def encode_image(self, img: Image.Image):
        if img.mode != "RGB":
            img = img.convert("RGB")

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            emb = self.model.get_image_features(**inputs)
            emb = F.normalize(emb, p=2, dim=-1).squeeze(0)      

        return emb.cpu().tolist()
