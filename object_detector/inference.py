from mmdet.apis import init_detector, inference_detector
from mmdet.core import DatasetEnum
import mmcv
import os
import json
from tqdm import tqdm
from pathlib import Path

config_file = 'projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.py'
checkpoint_file = 'models/co_dino_5scale_swin_large_16e_o365tococo.pth'

model = init_detector(config_file, checkpoint_file, DatasetEnum.COCO, device='cuda')
class_names = model.CLASSES

def predict_and_save(root_folder: Path, output_folder: Path):
    video_paths = sorted(root_folder.iterdir())

    for _video_path in tqdm(video_paths, desc="Processing videos"):
        _objects_in_vid = {}
        keyframes = list((_video_path / "keyframes").glob("*.webp"))
        for _image_path in tqdm(keyframes, desc=f"{_video_path.name}", leave=False):
            _obj_in_frame = []
            results = inference_detector(model, _image_path)
            for class_id, bboxes in enumerate(results):
                class_name = class_names[class_id]
                for box in bboxes:
                    if len(box) == 5:
                        x1, y1, x2, y2, score = map(float, box)
                        if score >= 0.3:
                            _obj_in_frame.append({
                                "catergory_id": class_id,
                                "bbox": [x1, y1, x2, y2],
                                "score": score,
                            })
            _objects_in_vid[_image_path.stem[9:]] = _obj_in_frame

        with open(output_folder / (_video_path.name + ".json"), 'w') as f:
            json.dump(_objects_in_vid, f, indent=4)

root_folder = Path('/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1')
output_folder = Path('/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/object_detector/json/full_batch1_codetr')
output_folder.mkdir(parents=True, exist_ok=True)

predict_and_save(root_folder, output_folder)
