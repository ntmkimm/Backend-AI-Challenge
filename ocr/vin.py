import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import json
import re
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
BATCH_SIZE = 1

# === Image Preprocessing ===
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = sorted(set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
        for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num),
        key=lambda x: x[0] * x[1]
    )
    def find_closest(aspect_ratio):
        best = (1, 1)
        diff = float('inf')
        for ratio in target_ratios:
            r = ratio[0] / ratio[1]
            d = abs(aspect_ratio - r)
            if d < diff:
                diff = d
                best = ratio
        return best

    best_ratio = find_closest(aspect_ratio)
    w, h = image_size * best_ratio[0], image_size * best_ratio[1]
    resized = image.resize((w, h))
    imgs = []
    for i in range(best_ratio[0] * best_ratio[1]):
        box = (
            (i % (w // image_size)) * image_size,
            (i // (w // image_size)) * image_size,
            ((i % (w // image_size)) + 1) * image_size,
            ((i // (w // image_size)) + 1) * image_size
        )
        imgs.append(resized.crop(box))
    if use_thumbnail and len(imgs) != 1:
        imgs.append(image.resize((image_size, image_size)))
    return imgs

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    tensor = torch.stack([transform(img) for img in images])
    return tensor

# === Dataset ===
class ImageDataset(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        return path.name, load_image(path, max_num=6)

# === Load Model ===
model = AutoModel.from_pretrained(
    "5CD-AI/Vintern-1B-v3_5",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    load_in_8bit=True,
    use_flash_attn=False,
)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)
model = model.eval().cuda()

chat_fn = model.module.chat if isinstance(model, torch.nn.DataParallel) else model.chat

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-1B-v3_5", trust_remote_code=True, use_fast=False)
print(tokenizer.eos_token, tokenizer.eos_token_id)
# === Main Execution ===
ROOT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1")

# có 363 video trong batch1
INDEXING = 363 // 2
# message = "first half"
message = "second half"
if (message == 'first half'): dirs = list(sorted(ROOT.iterdir()))[:INDEXING]
elif (message == 'second half'): dirs = list(sorted(ROOT.iterdir()))[INDEXING:]

for _dir in dirs:
    keyframe_dir = _dir / "keyframes"
    image_paths = sorted(list(keyframe_dir.glob("*.webp")))
    dataloader = DataLoader(ImageDataset(image_paths), batch_size=BATCH_SIZE, shuffle=False)
    
    dic = {}
    generation_config = dict(max_new_tokens=100, do_sample=False, num_beams=3, repetition_penalty=4.0)
    question = '<image>\nChỉ trích xuất thông tin dạng chữ và số trên cửa hiệu, vật thể trong ảnh mà không cung cấp thêm mô tả về ảnh.'
    # question = "<image>\nMô tả chi tiết bức ảnh này."
    for batch_names, batch_images in tqdm(dataloader, desc=f"Processing {message} {_dir.name}"):
        for name, pixel_values in zip(batch_names, batch_images):
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            response = chat_fn(tokenizer, pixel_values, question, generation_config, history=None, return_history=False)
            response = re.sub(r'\b\d{2}:\d{2}:\d{2}\b', '', response.strip().lower())
            dic[name[9:-5]] = response.replace("*", "").replace("\n", " ").replace("\r", " ")
            
    with open("ocr.json", "w") as f:
        json.dump(dic, f, indent=4)
