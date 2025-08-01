import cv2
import os
import torch
from PIL import Image
from tqdm import tqdm
import open_clip  # type: ignore 
import csv
import glob
import concurrent.futures
import torch.multiprocessing as mp
from queue import Empty

def preprocess_frame(frame, preprocess):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return preprocess(pil_image).unsqueeze(0)

def extract_features(frames, model, preprocess, device):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_frames = list(executor.map(lambda frame: preprocess_frame(frame, preprocess), frames))

    images = torch.cat(processed_frames).to(device)
    with torch.no_grad():
        features = model.encode_image(images)

    return features

def is_keyframe(curr_features, prev_features, curr_frame_id, prev_frame_id, frame_distance_threshold, clip_threshold, proximity_threshold, proximity_clip_threshold):
    clip_similarity = torch.sum(curr_features * prev_features) / (torch.norm(curr_features) * torch.norm(prev_features))
    frame_distance = curr_frame_id - prev_frame_id

    if frame_distance >= frame_distance_threshold:
        return True
    elif frame_distance < proximity_threshold:
        return clip_similarity < proximity_clip_threshold
    else:
        return clip_similarity < clip_threshold

import threading

save_lock = threading.Lock()

def save_image(img, path, quality=80, resize_factor=0.5):
    try:
        if img is None or img.size == 0:
            raise ValueError("Empty image")

        img_resized = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
        img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))

        with save_lock:  # prevent simultaneous write to same disk region
            img_pil.save(path, format="WEBP", quality=quality)

        # Double-check if file is written correctly
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise IOError(f"Failed to write image at {path}")

    except Exception as e:
        print(f"[save_image] Error saving image {path}: {e}")

def process_video(video_path, output_folder, maps_folder, clip_threshold, frame_distance_threshold, proximity_threshold, proximity_clip_threshold, batch_size, sample_rate, skip_frames, device_id):
    os.makedirs(output_folder, exist_ok=True)
    keyframes_folder = os.path.join(output_folder, "keyframes")
    os.makedirs(keyframes_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
    model = model.to(device).eval()

    frames = []
    frame_indices = []
    keyframes = []
    prev_features = None
    prev_keyframe_id = -frame_distance_threshold

    pbar = tqdm(total=total_frames, desc=f"[GPU {device_id}] Processing {os.path.basename(video_path)}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    map_file_path = os.path.join(maps_folder, f"{video_name}_map.csv")

    with open(map_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame ID', 'Seconds'])

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                break

            if (frame_count % (fps // sample_rate) == 0) and (frame_count // (fps // sample_rate)) % (skip_frames + 1) == 0:
                frames.append(frame)
                frame_indices.append(frame_count)

                if len(frames) == batch_size:
                    features = extract_features(frames, model, preprocess, device)

                    for i, (frame, feature) in enumerate(zip(frames, features)):
                        frame_id = frame_indices[i]
                        if prev_features is None or is_keyframe(feature, prev_features, frame_id, prev_keyframe_id, frame_distance_threshold, clip_threshold, proximity_threshold, proximity_clip_threshold):
                            keyframes.append((frame, feature))
                            save_image(frame, os.path.join(keyframes_folder, f"keyframe_{frame_id}.webp"), quality=80, resize_factor=0.5)
                            seconds = frame_id / fps
                            csv_writer.writerow([frame_id, f"{seconds:.2f}"])
                            prev_features = feature
                            prev_keyframe_id = frame_id

                    frames = []
                    frame_indices = []

            frame_count += 1
            pbar.update(1)

        if frames:
            features = extract_features(frames, model, preprocess, device)

            for i, (frame, feature) in enumerate(zip(frames, features)):
                frame_id = frame_indices[i]
                if prev_features is None or is_keyframe(feature, prev_features, frame_id, prev_keyframe_id, frame_distance_threshold, clip_threshold, proximity_threshold, proximity_clip_threshold):
                    keyframes.append((frame, feature))
                    save_image(frame, os.path.join(keyframes_folder, f"keyframe_{frame_id}.webp"), quality=80, resize_factor=0.5)
                    seconds = frame_id / fps
                    csv_writer.writerow([frame_id, f"{seconds:.2f}"])
                    prev_features = feature
                    prev_keyframe_id = frame_id

    cap.release()
    pbar.close()

    if os.path.exists(keyframes_folder) and not os.listdir(keyframes_folder):
        os.rmdir(keyframes_folder)
        os.rmdir(output_folder)

    return keyframes

def process_all_videos_worker(video_queue, output_base_folder, clip_threshold, frame_distance_threshold, proximity_threshold, proximity_clip_threshold, batch_size, sample_rate, skip_frames, device_id):
    while True:
        try:
            video_path = video_queue.get(timeout=3)
        except Empty:
            break

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_folder = os.path.join(output_base_folder, video_name)
        maps_folder = os.path.join(output_base_folder, "maps")
        os.makedirs(maps_folder, exist_ok=True)

        print(f"[GPU {device_id}] Starting video: {video_name}")
        try:
            keyframes = process_video(video_path, output_folder, maps_folder,
                                      clip_threshold, frame_distance_threshold, proximity_threshold, proximity_clip_threshold,
                                      batch_size, sample_rate, skip_frames, device_id)
            print(f"[GPU {device_id}] Finished video: {video_name}, keyframes: {len(keyframes)}")
        except Exception as e:
            print(f"[GPU {device_id}] Error processing {video_name}: {str(e)}")

if __name__ == "__main__":
    mp.set_start_method('spawn')

    input_folder = '/mlcv2/Datasets/HCMAI24/updated/videos/batch1'
    output_base_folder = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1'
    video_files = sorted(glob.glob(os.path.join(input_folder, '*.mp4')))

    num_gpus = torch.cuda.device_count()

    clip_threshold = 0.96
    frame_distance_threshold = 75
    proximity_threshold = 15
    proximity_clip_threshold = 0.80
    batch_size = 8
    sample_rate = 25
    skip_frames = 5

    video_queue = mp.Queue()
    for vf in video_files:
        video_queue.put(vf)

    processes = []
    for device_id in range(num_gpus):
        p = mp.Process(target=process_all_videos_worker, args=(
            video_queue, output_base_folder,
            clip_threshold, frame_distance_threshold,
            proximity_threshold, proximity_clip_threshold,
            batch_size, sample_rate, skip_frames, device_id))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All videos processed.")
