# from pathlib import Path
# from tqdm import tqdm

# root = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/supplement")

# start = "K01_V001"
# end = "L31_V001"

# _videos = []
# for _video in sorted(root.iterdir()):
#     # if _video.name >= start and _video.name < end:
#         _videos.append(_video)

# prev = 0
# count = 0
# heh = []
# for _video in tqdm(_videos):
#     keyframes_folder = _video / "keyframes"
#     if not keyframes_folder.exists():
#         print(_video)
#     kf_nums = []
#     for kf in keyframes_folder.glob("*.webp"):
#         kf_nums.append(int(kf.stem[9:]))
#     kf_nums.sort()
#     for now in kf_nums:
#         if now - prev >= 90:
#             print(f"{_video.name}: {now} - {prev} = {now - prev}")
#             count += 1
#             heh.append(_video.name)
#             break
#         prev = now

# print(heh)
# print(count)

            
import os
import av

def count_videos_and_hours(folder_path):
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm')
    total_duration = 0.0
    video_count = 0

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)
                try:
                    container = av.open(video_path)
                    duration = float(container.duration) / 1_000_000  # microseconds → seconds
                    total_duration += duration
                    video_count += 1
                    container.close()
                except Exception as e:
                    print(f"Skipping {file} due to error: {e}")

    total_hours = total_duration / 3600
    print(f"Total videos: {video_count}")
    print(f"Total duration: {total_hours:.2f} hours")

# Example usage
count_videos_and_hours("/mlcv1/Datasets/HCMAI25/full")
