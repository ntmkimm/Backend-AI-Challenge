from pathlib import Path
from tqdm import tqdm

root = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/supplement")

start = "K01_V001"
end = "L31_V001"

_videos = []
for _video in sorted(root.iterdir()):
    # if _video.name >= start and _video.name < end:
        _videos.append(_video)

prev = 0
count = 0
heh = []
for _video in tqdm(_videos):
    keyframes_folder = _video / "keyframes"
    if not keyframes_folder.exists():
        print(_video)
    kf_nums = []
    for kf in keyframes_folder.glob("*.webp"):
        kf_nums.append(int(kf.stem[9:]))
    kf_nums.sort()
    for now in kf_nums:
        if now - prev >= 90:
            print(f"{_video.name}: {now} - {prev} = {now - prev}")
            count += 1
            heh.append(_video.name)
            break
        prev = now

print(heh)
print(count)

            
