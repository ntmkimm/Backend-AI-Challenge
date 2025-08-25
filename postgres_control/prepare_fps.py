import cv2
import os
import json

def extract_fps_from_videos(video_folder, output_json="video_fps.json"):
    fps_dict = {}

    # Loop through all files in the folder
    for file_name in os.listdir(video_folder):
        file_path = os.path.join(video_folder, file_name)

        # Skip if not a file or not a video format
        if not os.path.isfile(file_path) or not file_name.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        # Open video with OpenCV
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"⚠️ Could not open {file_name}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_dict[file_name] = fps
        cap.release()

    # Save results to JSON
    with open(output_json, "w") as f:
        json.dump(fps_dict, f, indent=4)

    print(f"✅ FPS data saved to {output_json}")


if __name__ == "__main__":
    video_folder = "/mlcv2/Datasets/HCMAI25/batch1/video"  # change to your dataset path
    extract_fps_from_videos(video_folder)
