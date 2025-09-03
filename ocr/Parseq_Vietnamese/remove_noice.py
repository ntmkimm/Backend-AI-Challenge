import json
from pathlib import Path
import tqdm

class Args:
    checkpoint = 'new-parseq.ckpt'
    root_videos = Path('/mlcv2/Datasets/HCMAI25/batch2/video')
    root_bboxes = Path('/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/ocr/json/batch2_2025_parseq')
    output = Path('/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge')
    device = 'cuda'
    batch_size = 4
    num_workers = 1
    rotation = 0

args = Args()

def remove_bboxes_in_y_range(infos, lo_bound, up_bound):
    return [info for info in infos if not (lo_bound <= info['bbox'][1] <= up_bound and
                                           lo_bound <= info['bbox'][3] <= up_bound)]

# --- Video Range Config ---

start_video = 'K01_V001' # include this video
end_video = 'K21_V001' # not include this video

print("start_video: ", start_video)
print("end_video: ", end_video)

import time
video_files = []
for _video_path in sorted(args.root_videos.glob("*.mp4")):
    video_name = _video_path.stem
    if not (start_video <= video_name < end_video):
        continue
    video_files.append(_video_path)


for _video_path in tqdm.tqdm(video_files, desc="Overall Progress"):
    _video_id = _video_path.stem
    with open(args.root_bboxes / (_video_id + ".json"), "r", encoding='utf-8') as fi:
        _datas = json.load(fi)

    print(f"\nProcessing video: {_video_id}") 
    output_file = args.output / _video_id / "ocr_parseq.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # if output_file.exists():
    #     continue
    _video_res = {}
    for _frame_id, _infos in _datas.items():
        _infos = remove_bboxes_in_y_range(_infos, lo_bound=980, up_bound=1040)
        text = " ".join([info.get("text", "") for info in _infos])
        _video_res[_frame_id] = text
        # print(f"{_video_id} - {_frame_id}: {text}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(_video_res, f, indent=4, ensure_ascii=False)

print("✅ All videos processed!")