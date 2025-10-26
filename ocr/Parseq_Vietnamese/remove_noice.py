import json
from pathlib import Path
import tqdm

class Args:
    root_videos = Path('/mlcv1/Datasets/HCMAI25/full')
    root_bboxes = Path('/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/ocr/json/supplement_newmodel_parseq_new')
    output = Path('/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge')
    device = 'cuda'
    batch_size = 4
    num_workers = 1
    rotation = 0

args = Args()

def remove_bboxes_in_region(
    infos,
    y_range=None,                # (lo_y, up_y)
    rect=None                    # (x_min, y_min, x_max, y_max)
):
    """
    Remove bounding boxes that fall inside a specified vertical range or rectangular region.

    Args:
        infos (list[dict]): List of OCR/ASR info with 'bbox' field = [x1, y1, x2, y2].
        y_range (tuple): (lo_y, up_y) to remove boxes within this vertical band.
        rect (tuple): (x_min, y_min, x_max, y_max) to remove boxes inside this rectangle.

    Returns:
        list[dict]: Filtered infos (boxes outside the specified range/region).
    """
    results = []
    for info in infos:
        x1, y1, x2, y2 = info['bbox']
        remove = False

        # --- remove if inside Y range ---
        if y_range is not None:
            lo_y, up_y = y_range
            if lo_y <= y1 <= up_y and lo_y <= y2 <= up_y:
                remove = True

        # --- remove if inside rectangular region ---
        if rect is not None:
            x_min, y_min, x_max, y_max = rect
            if (x_min <= x1 <= x_max and x_min <= x2 <= x_max and
                y_min <= y1 <= y_max and y_min <= y2 <= y_max):
                remove = True

        if not remove:
            results.append(info)

    return results

# --- Video Range Config ---

start_video = 'K01_V001' # include this video
end_video = 'L23_V001' # not include this video

print("start_video: ", start_video)
print("end_video: ", end_video)

import time
video_files = []
for _video_path in sorted(args.root_videos.glob("*.mp4")):
    video_name = _video_path.stem
    if not (start_video <= video_name < end_video):
        continue
    video_files.append(_video_path)


# for _video_path in tqdm.tqdm(video_files, desc="Overall Progress"):
#     _video_id = _video_path.stem
#     with open(args.root_bboxes / (_video_id + ".json"), "r", encoding='utf-8') as fi:
#         _datas = json.load(fi)

#     print(f"\nProcessing video: {_video_id}") 
#     output_file = args.output / _video_id / "ocr_parseq_newmodel.json"
#     output_file.parent.mkdir(parents=True, exist_ok=True)
    
#     # if output_file.exists():
#     #     continue
#     _video_res = {}
#     for _frame_id, _infos in _datas.items():
#         # K
#         if "K" in _video_id:
#             _infos = remove_bboxes_in_region(_infos, y_range=(980, 1040), rect=(1560, 130, 1710, 180))
#             _infos = remove_bboxes_in_region(_infos, rect=(1575, 85, 1755, 160)) # HTV7
#             _infos = remove_bboxes_in_region(_infos, rect=(1730, 105, 1790, 165)) # HD
        
#         if "L" in _video_id:
#             _infos = remove_bboxes_in_region(_infos, y_range=(655, 690), rect=(1040, 80, 1144, 125))
#             _infos = remove_bboxes_in_region(_infos, rect=(1050, 50, 1170, 110))
#             _infos = remove_bboxes_in_region(_infos, rect=(1150, 65, 1199, 112))
        
#         text = " ".join([info.get("text", "") for info in _infos])
#         _video_res[_frame_id] = text
#         # print(f"{_video_id} - {_frame_id}: {text}")

#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump(_video_res, f, indent=4, ensure_ascii=False)

# suyra
suyra_folder = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/ocr/suyra/L25_OCR")
for _video_path in tqdm.tqdm(sorted(suyra_folder).iterdir(), desc="Overall Progress"):
    _video_id = _video_path.stem
    _video_res = {}
    for _txt in _video_path.glob("*.txt"): 
        _frame_id = _txt.stem
        with open(_txt, "r") as f:
            texts = f.readlines()
            text = " ".join(line.strip() for line in texts).lower()
            _video_res[_frame_id] = text

    print(f"\nProcessing video: {_video_id}") 
    output_file = args.output / _video_id / "ocr_parseq_newmodel.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(_video_res, f, indent=4, ensure_ascii=False)
    

print("✅ All videos processed!")