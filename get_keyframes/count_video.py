from pathlib import Path

ROOT = Path('/mlcv2/Datasets/HCMAI24/updated/videos/batch1')

print(len(list(ROOT.glob("*.mp4"))))

# 363

OUT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1")
print(len(list(OUT.iterdir()))) 


# opencv-python 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 1.26.4 which is incompatible.
# opencv-python-headless 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= "3.9", but you have numpy 1.26.4 which is incompatible.