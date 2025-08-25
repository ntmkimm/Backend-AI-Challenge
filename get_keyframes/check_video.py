from pathlib import Path
import cv2
from tqdm import tqdm

root_videos = Path("/mlcv2/Datasets/HCMAI25/batch1/video")
map_folder = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/batch1")

problem_videos = []

def check_video_process_all_frame_ids(video_path: Path, input_folder: Path, output_base_folder: Path):
    video_name = video_path.stem 
    if not (output_base_folder / video_name / "keyframes").exists():
        return False
    keyframe_paths = (output_base_folder / video_name / "keyframes").glob("*.webp")
    if not keyframe_paths:
        return False
    last_frame_id = 0 
    for _keyframe in tqdm(keyframe_paths): 
        last_frame_id = max(last_frame_id, int(_keyframe.stem[9:]))
    
    cap = cv2.VideoCapture(str(_video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames - last_frame_id > 25 * 60:
        print("problem: ", video_name, " distant ", total_frames - last_frame_id)
        return False
    return True

for _video_path in tqdm(sorted(root_videos.glob("*.mp4"))[::-1]):
    video_name = _video_path.stem
    if not (video_name == 'L26_V336' or video_name == 'L26_V074'): continue
    
    if not (map_folder / video_name / "keyframes").exists():
        continue
    keyframe_paths = (map_folder / video_name / "keyframes").glob("*.webp")
    if not keyframe_paths:
        continue

    last_frame_id = 0 
    for _keyframe in tqdm(keyframe_paths): 
        last_frame_id = max(last_frame_id, int(_keyframe.stem[9:]))
    
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

        