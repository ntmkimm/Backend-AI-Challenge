import csv
import subprocess
import tempfile
import json
from pathlib import Path
from tqdm import tqdm  

DATASET = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1")
ROOT = Path("/mlcv2/Datasets/HCMAI24/streaming/batch1_audio/")

# Iterate over each video directory
for _vid in ROOT.iterdir():
    video_id = _vid.stem
    output_path = DATASET / video_id / "asr.json"

    if output_path.exists():
        output_path.unlink()  # Delete the file
        print(f"Deleted: {output_path}")
    else:
        print(f"Not found: {output_path}")
