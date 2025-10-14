import csv
import subprocess
import tempfile
import json
from pathlib import Path
from tqdm import tqdm  

DATASET = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge")
ROOT = Path("/mlcv1/Datasets/HCMAI25/video")

start_video = "K01_V001"
end_video = "L01_V001"

# Iterate over each video directory
for _vid in tqdm(DATASET.iterdir()):
    video_id = _vid.stem
    if not (start_video <= video_id and video_id < end_video): continue
    output_path = DATASET / video_id / "ocr_parseq_new.json"

    if output_path.exists():
        output_path.unlink()  # Delete the file
        print(f"Deleted: {output_path}")
    # else:
    #     print(f"Not found: {output_path}")
