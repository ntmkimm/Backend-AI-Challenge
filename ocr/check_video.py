from pathlib import Path
import cv2
from tqdm import tqdm
import json

root_videos = Path("/mlcv2/Datasets/HCMAI25/batch1/video")
map_folder = Path("./json/batch1_2025")

problem_videos = []

for _video_path in tqdm(sorted(root_videos.glob("*.mp4"))):
    video_name = _video_path.stem
    if not (map_folder / (video_name + ".json")).exists():
        problem_videos.append(video_name)
        continue
    with open(map_folder / (video_name + ".json"), "r") as f:
        _dic = json.load(f)

    last_frame_id = 0 
    for _keyframe in _dic.keys(): 
        last_frame_id = max(last_frame_id, int(_keyframe))
    
    cap = cv2.VideoCapture(str(_video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # You can adjust this threshold as needed
    if total_frames - last_frame_id > 25 * 60:
        print("problem: ", video_name, " distant ", total_frames - last_frame_id)
        problem_videos.append(video_name)

# Write to txt file
output_txt_path = Path("videos_with_keyframe_not_near_end.txt")
with open(output_txt_path, "w") as f:
    for video in problem_videos:
        f.write(video + "\n")

print(f"Found {len(problem_videos)} video(s) with last keyframe far from end.")

        