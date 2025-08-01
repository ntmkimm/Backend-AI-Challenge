import torch
import torch.nn.functional as F
import open_clip
from typing import List
from config.settings import CLIP_MODEL, DEVICE, BATCH_SIZE
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image
import asyncio

class CLIPService:
    def __init__(self):
        print("Init Clip Service...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained="dfn5b")
        self.tokenizer = open_clip.get_tokenizer(CLIP_MODEL)
        self.model = self.model.to(self.device).eval()

    def encode_text_batch(self, texts: List[str], batch_size: int = BATCH_SIZE):
        """
        Encode a batch of texts to CLIP embeddings.
        Args:
            texts (List[str]): List of input strings.
            batch_size (int): Batch size for encoding.
        Returns:
            embeddings (List[List[float]]): List of embeddings (each is a list of floats).
        """
        if not texts:
            return []

        embeddings_list = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            with torch.no_grad():
                tokens = self.tokenizer(batch_texts).to(self.device)
                batch_embeddings = self.model.encode_text(tokens)
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=-1)
                embeddings_list.append(batch_embeddings.cpu())

        embeddings = torch.cat(embeddings_list, dim=0)
        return embeddings.tolist() 

    def encode_single_text(self, text: str):
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            embedding = self.model.encode_text(tokens)
            return F.normalize(embedding, p=2, dim=-1).cpu().tolist()
        
    def encode_image(self, image: Image.Image):
        """
        Encode a single PIL Image (RGB) into CLIP embedding.
        Args:
            image (PIL.Image.Image): Ảnh input dạng PIL Image (nên là RGB).
        Returns:
            embedding (list[float]): embedding đã chuẩn hóa L2.
        """
        # Đảm bảo là PIL Image RGB
        if not isinstance(image, Image.Image):
            raise TypeError(f"Input must be PIL.Image.Image, got {type(image)}")
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)  # [1, 3, H, W]

        with torch.no_grad():
            image_features = self.model.encode_image(img_tensor)
            image_features = F.normalize(image_features, p=2, dim=-1)
            return image_features.squeeze(0).cpu().tolist()  # [D]

    def encode_image_batch(self, images: List[Image.Image]):
        """
        Encode a batch of PIL Images (RGB) into CLIP embeddings.
        Args:
            images (List[PIL.Image.Image]): Danh sách ảnh PIL Image (nên là RGB).
        Returns:
            embeddings (np.ndarray): (N, D), embedding đã chuẩn hóa L2.
        """
        # Preprocess toàn bộ ảnh thành tensor batch
        processed_tensors = []
        for img in images:
            if not isinstance(img, Image.Image):
                raise TypeError(f"Input must be PIL.Image.Image, got {type(img)}")
            if img.mode != "RGB":
                img = img.convert("RGB")
            processed_tensors.append(self.preprocess(img))
        if not processed_tensors:
            return np.empty((0, self.model.visual.output_dim))
        batch_tensor = torch.stack(processed_tensors, dim=0).to(self.device)  # (N, 3, H, W)

        with torch.no_grad():
            features = self.model.encode_image(batch_tensor)
            features = F.normalize(features, p=2, dim=-1)  # (N, D)
            return features.cpu().numpy()  # (N, D)
        
    async def encode_text_batch_async(self, texts: List[str], batch_size: int = BATCH_SIZE):
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, lambda: self.encode_text_batch(texts, batch_size))
        return embeddings

    async def encode_image_async(self, image: Image.Image):
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, lambda: self.encode_image(image))
        return embedding
