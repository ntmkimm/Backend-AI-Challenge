import torch
from PIL import Image
from pathlib import Path
from transformers import BlipForConditionalGeneration, BlipProcessor
from transformers import Blip2ForConditionalGeneration, Blip2Processor
from transformers import AutoProcessor, Blip2Model
import json
from tqdm import tqdm

# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained(
#     "Salesforce/blip-image-captioning-base",
#     torch_dtype=torch.float32
# )

# model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

ROOT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1")

dirs = list(sorted(ROOT.iterdir()))

BATCH_SIZE = 48  
import time
start_time = time.time()
for _dir in tqdm(dirs):
    keyframe_dir = _dir / "keyframes"
    image_paths = sorted(list(keyframe_dir.glob("*.webp")))
    dic = {}

    # Batch loop
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=f"{_dir.name} "):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
        batch_inputs = processor(batch_images, return_tensors="pt").to(device, torch.float32)
        with torch.no_grad():
            image_outputs = model.get_image_features(**batch_inputs) 
            # Print the shape of the image features tensor
            print(image_outputs.pooler_output.shape)
            print(image_outputs.last_hidden_state.shape)
            # generated_ids = model.generate(**batch_inputs, max_new_tokens=100)
            # print(generated_ids.shape)
            # batch_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
            # for path, caption in zip(batch_paths, batch_captions):
            #     frame_id = path.stem[9:]
            #     dic[frame_id] = caption.strip()
            #     print(caption)
        break 
    break

    # Save captions (uncomment to save per video)
    with open(_dir / "scene_vi_blip.json", "w", encoding='utf-8') as f:
        json.dump(dic, f, indent=4)
        
end_time = time.time()
print("time process: ", end_time - start_time)