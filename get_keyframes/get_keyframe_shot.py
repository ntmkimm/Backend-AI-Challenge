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
import threading
import numpy as np
from pathlib import Path
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

def read_shot_boundaries(scenes_file_path):
    """Reads the shot boundaries from a .scenes.txt file."""
    shots = []
    with open(scenes_file_path, 'r') as f:
        for line in f:
            start_frame, end_frame = map(int, line.strip().split())
            shots.append((start_frame, end_frame))
    return shots

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
    if prev_features is None:
        return True

    clip_similarity = torch.sum(curr_features * prev_features) / (torch.norm(curr_features) * torch.norm(prev_features))
    frame_distance = curr_frame_id - prev_frame_id

    if frame_distance >= frame_distance_threshold:
        return True
    elif frame_distance < proximity_threshold:
        return clip_similarity < proximity_clip_threshold
    else:
        return clip_similarity < clip_threshold

save_lock = threading.Lock()

def save_image(img, path, quality=80, resize_factor=0.5):
    try:
        if img is None or img.size == 0:
            raise ValueError("Empty image")

        img_resized = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
        img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))

        with save_lock:  # prevent simultaneous write to same disk region
            img_pil.save(path, format="WEBP", quality=quality)

        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise IOError(f"Failed to write image at {path}")

    except Exception as e:
        print(f"[save_image] Error saving image {path}: {e}")

def process_video(video_path, scenes_file_path, output_folder, maps_folder, clip_threshold, frame_distance_threshold, proximity_threshold, proximity_clip_threshold, batch_size, sample_rate, skip_frames, device_id):
    os.makedirs(output_folder, exist_ok=True)
    keyframes_folder = os.path.join(output_folder, "keyframes")
    os.makedirs(keyframes_folder, exist_ok=True)
    vector_folder = os.path.join(output_folder, "vector_file")
    os.makedirs(vector_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
    model = model.to(device).eval()

    shot_boundaries = read_shot_boundaries(scenes_file_path)
    total_frames_to_process = sum(end - start for start, end in shot_boundaries)

    pbar = tqdm(total=total_frames_to_process, desc=f"[GPU {device_id}] Processing {os.path.basename(video_path)}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    map_file_path = os.path.join(maps_folder, f"{video_name}_map.csv")

    all_keyframes = []

    with open(map_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame ID', 'Seconds'])

        for start_frame, end_frame in shot_boundaries:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames = []
            frame_indices = []
            prev_features = None
            prev_keyframe_id = -frame_distance_threshold

            for frame_count in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    break

                if (frame_count % (fps // sample_rate) == 0) and (frame_count // (fps // sample_rate)) % (skip_frames + 1) == 0:
                    frames.append(frame)
                    frame_indices.append(frame_count)

                    if len(frames) == batch_size:
                        features_batch = extract_features(frames, model, preprocess, device)

                        for i, (frame_in_batch, feature) in enumerate(zip(frames, features_batch)):
                            frame_id = frame_indices[i]
                            if is_keyframe(feature, prev_features, frame_id, prev_keyframe_id, frame_distance_threshold, clip_threshold, proximity_threshold, proximity_clip_threshold):
                                all_keyframes.append((frame_in_batch, feature))
                                save_image(frame_in_batch, os.path.join(keyframes_folder, f"keyframe_{frame_id}.webp"), quality=80, resize_factor=0.5)

                                # Save vector to .npz file
                                vector_path = os.path.join(vector_folder, f"keyframe_{frame_id}.npz")
                                np.savez_compressed(vector_path, feature=feature.cpu().numpy())

                                seconds = frame_id / fps
                                csv_writer.writerow([frame_id, f"{seconds:.2f}"])
                                prev_features = feature
                                prev_keyframe_id = frame_id

                        frames = []
                        frame_indices = []

                pbar.update(1)

            if frames:
                features_batch = extract_features(frames, model, preprocess, device)
                for i, (frame_in_batch, feature) in enumerate(zip(frames, features_batch)):
                    frame_id = frame_indices[i]
                    if is_keyframe(feature, prev_features, frame_id, prev_keyframe_id, frame_distance_threshold, clip_threshold, proximity_threshold, proximity_clip_threshold):
                        all_keyframes.append((frame_in_batch, feature))
                        save_image(frame_in_batch, os.path.join(keyframes_folder, f"keyframe_{frame_id}.webp"), quality=80, resize_factor=0.5)

                        # Save vector to .npz file
                        vector_path = os.path.join(vector_folder, f"keyframe_{frame_id}.npz")
                        np.savez_compressed(vector_path, feature=feature.cpu().numpy())

                        seconds = frame_id / fps
                        csv_writer.writerow([frame_id, f"{seconds:.2f}"])
                        prev_features = feature
                        prev_keyframe_id = frame_id

    cap.release()
    pbar.close()

    if os.path.exists(keyframes_folder) and not os.listdir(keyframes_folder):
        os.rmdir(keyframes_folder)
        os.rmdir(vector_folder)
        os.rmdir(output_folder)

    return all_keyframes

def extract_video_id_from_scenes_file(scenes_file_path):
    """Extract video ID from scenes file name (e.g., L21_V001.scenes.txt -> L21_V001)"""
    filename = os.path.basename(scenes_file_path)
    # Remove .scenes.txt extension
    if filename.endswith('.scenes.txt'):
        return filename[:-11]
    elif filename.endswith('.mp4.scenes.txt'):
        return filename[:-15]
    else:
        return filename.split('.')[0]

def get_video_path_from_id(video_id, input_folder):
    """Find the corresponding video file for a given video ID"""
    # Try different extensions
    extensions = ['.mp4', '.avi', '.mov', '.mkv']
    for ext in extensions:
        video_path = os.path.join(input_folder, f"{video_id}{ext}")
        if os.path.exists(video_path):
            return video_path
    return None

def process_all_videos_worker(video_queue, shot_folder, input_folder, output_base_folder, 
                             clip_threshold, frame_distance_threshold, proximity_threshold, 
                             proximity_clip_threshold, batch_size, sample_rate, skip_frames, device_id, stop_event):
    while not stop_event.is_set():
        try:
            scenes_file_path = video_queue.get(timeout=1)
        except Empty:
            continue  # Keep checking for new files instead of breaking

        if scenes_file_path is None:  # Shutdown signal
            break

        video_id = extract_video_id_from_scenes_file(scenes_file_path)
        video_path = get_video_path_from_id(video_id, input_folder)

        if not video_path:
            print(f"[GPU {device_id}] Video file not found for {video_id}, skipping.")
            continue

        if not os.path.exists(scenes_file_path):
            print(f"[GPU {device_id}] Scenes file not found for {video_id}, skipping.")
            continue

        output_folder = os.path.join(output_base_folder, video_id)
        maps_folder = os.path.join(output_base_folder, "maps")
        os.makedirs(maps_folder, exist_ok=True)

        print(f"[GPU {device_id}] Starting video: {video_id}")
        try:
            keyframes = process_video(video_path, scenes_file_path, output_folder, maps_folder,
                                      clip_threshold, frame_distance_threshold,
                                      proximity_threshold, proximity_clip_threshold,
                                      batch_size, sample_rate, skip_frames, device_id)
            print(f"[GPU {device_id}] Finished video: {video_id}, keyframes: {len(keyframes)}")
        except Exception as e:
            print(f"[GPU {device_id}] Error processing {video_id}: {str(e)}")

class ScenesFileHandler(FileSystemEventHandler):
    def __init__(self, video_queue, processed_files, lock):
        self.video_queue = video_queue
        self.processed_files = processed_files
        self.lock = lock

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.scenes.txt'):
            # Wait a moment for the file to be fully written
            time.sleep(1)
            with self.lock:
                if event.src_path not in self.processed_files:
                    print(f"New scenes file detected: {event.src_path}")
                    self.video_queue.put(event.src_path)
                    self.processed_files[event.src_path] = True

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.scenes.txt'):
            # Wait a moment for the file to be fully written
            time.sleep(1)
            with self.lock:
                if event.src_path not in self.processed_files:
                    print(f"Scenes file modified: {event.src_path}")
                    self.video_queue.put(event.src_path)
                    self.processed_files[event.src_path] = True

def start_file_watcher(shot_folder, video_queue, processed_files, lock):
    """Start watching the shot folder for new .scenes.txt files"""
    event_handler = ScenesFileHandler(video_queue, processed_files, lock)
    observer = Observer()
    observer.schedule(event_handler, shot_folder, recursive=False)
    observer.start()
    print(f"File watcher started for folder: {shot_folder}")
    return observer

if __name__ == "__main__":
    mp.set_start_method('spawn')

    input_folder = '/mlcv2/Datasets/HCMAI25/batch1/video'
    shot_folder = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/shot_batch1'
    output_base_folder = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/batch1'

    num_gpus = torch.cuda.device_count()

    clip_threshold = 0.96
    frame_distance_threshold = 75
    proximity_threshold = 15
    proximity_clip_threshold = 0.80
    batch_size = 8
    sample_rate = 25
    skip_frames = 5

    # Create shared queue, processed files set, and lock
    video_queue = mp.Queue()
    manager = mp.Manager()
    processed_files = manager.dict()
    files_lock = manager.Lock()
    stop_event = manager.Event()

    # First, process all existing .scenes.txt files
    existing_scenes_files = sorted(glob.glob(os.path.join(shot_folder, '*.scenes.txt')))
    print(f"Found {len(existing_scenes_files)} existing scenes files to process")
    
    with files_lock:
        for scenes_file in existing_scenes_files:
            video_queue.put(scenes_file)
            processed_files[scenes_file] = True

    # Start file watcher for new files
    observer = start_file_watcher(shot_folder, video_queue, processed_files, files_lock)

    # Start worker processes
    processes = []
    for device_id in range(num_gpus):
        p = mp.Process(target=process_all_videos_worker, args=(
            video_queue, shot_folder, input_folder, output_base_folder,
            clip_threshold, frame_distance_threshold,
            proximity_threshold, proximity_clip_threshold,
            batch_size, sample_rate, skip_frames, device_id, stop_event))
        p.start()
        processes.append(p)

    try:
        print("Processing existing files and watching for new ones...")
        print("Press Ctrl+C to stop the file watcher and exit")
        
        # Monitor progress
        processed_count = 0
        last_queue_size = video_queue.qsize()
        
        while True:
            time.sleep(5)  # Check every 5 seconds
            current_queue_size = video_queue.qsize()
            
            if current_queue_size != last_queue_size:
                print(f"Queue size: {current_queue_size}, Total files tracked: {len(processed_files)}")
                last_queue_size = current_queue_size
                
            # Check if any process is still alive
            alive_processes = [p for p in processes if p.is_alive()]
            if not alive_processes and video_queue.empty():
                print("All workers finished and queue is empty.")
                break
            
    except KeyboardInterrupt:
        print("\nShutting down...")
        stop_event.set()
        observer.stop()
        
        # Send shutdown signals to workers
        for _ in range(num_gpus):
            video_queue.put(None)
        
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        observer.join()
        print("All processes have been terminated.")