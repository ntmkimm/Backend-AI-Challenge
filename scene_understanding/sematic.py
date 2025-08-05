import torch
from PIL import Image
from pathlib import Path
from transformers import BlipForConditionalGeneration, BlipProcessor
import json
from tqdm import tqdm

# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained(
#     "Salesforce/blip-image-captioning-base",
#     torch_dtype=torch.float32
# )

from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16, device_map="auto")


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

ROOT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1")
INDEXING = 363 // 2

dirs = list(sorted(ROOT.iterdir()))

BATCH_SIZE = 48  # You can try 16, 32, or higher if enough VRAM

for _dir in tqdm(dirs):
    keyframe_dir = _dir / "keyframes"
    image_paths = sorted(list(keyframe_dir.glob("*.webp")))
    dic = {}

    # Batch loop
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=f"{_dir.name} "):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
        batch_inputs = processor(batch_images, return_tensors="pt").to(device, torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(**batch_inputs, max_new_tokens=100)
            batch_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for path, caption in zip(batch_paths, batch_captions):
                frame_id = path.stem[9:]
                dic[frame_id] = caption.strip()
                print(caption)

    # Save captions (uncomment to save per video)
    with open(_dir / "scene_vi_blip2.json", "w", encoding='utf-8') as f:
        json.dump(dic, f, indent=4)

