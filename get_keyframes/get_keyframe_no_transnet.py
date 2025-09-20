import cv2
import os
import torch
from PIL import Image
from tqdm import tqdm
import open_clip  # type: ignore
import glob
import concurrent.futures
import torch.multiprocessing as mp
from queue import Empty
import threading
import numpy as np
from pathlib import Path
from services.MetaCLIP.src.mini_clip.factory import create_model_and_transforms, get_tokenizer


# def preprocess_frame(frame, preprocess):
#     pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     return preprocess(pil_image).unsqueeze(0)
def preprocess_frame(frame, preprocess):
    pil_image = Image.fromarray(frame) # Frame is already RGB
    return preprocess(pil_image).unsqueeze(0)


def extract_features(frames, model, preprocess, device):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_frames = list(executor.map(lambda frame: preprocess_frame(frame, preprocess), frames))
    images = torch.cat(processed_frames).to(device)
    with torch.no_grad():
        features = model.encode_image(images)
    return features


def is_keyframe(curr_features, prev_features, curr_frame_id, prev_frame_id,
                frame_distance_threshold, clip_threshold, proximity_threshold=None, proximity_clip_threshold=None):
    if prev_features is None:
        return True
    clip_similarity = torch.sum(curr_features * prev_features) / (torch.norm(curr_features) * torch.norm(prev_features))
    frame_distance = int(curr_frame_id) - int(prev_frame_id)
    if frame_distance >= frame_distance_threshold:
        return True
    elif proximity_clip_threshold and proximity_threshold and frame_distance < proximity_threshold:
        return clip_similarity < proximity_clip_threshold
    else:
        return clip_similarity < clip_threshold

save_lock = threading.Lock()

def save_image(img, path, quality=80, resize_factor=0.5):
    try:
        if img is None or img.size == 0:
            raise ValueError("Empty image")
        img_resized = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
        # img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        img_pil = Image.fromarray(img_resized)
        with save_lock:
            img_pil.save(path, format="WEBP", quality=quality)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise IOError(f"Failed to write image at {path}")
    except Exception as e:
        print(f"[save_image] Error saving image {path}: {e}")

import av  # New importimport av

def process_video(video_path, output_folder,
                  clip_threshold, frame_distance_threshold, proximity_threshold, proximity_clip_threshold,
                  batch_size, sample_rate, skip_frames, device_id):

    os.makedirs(output_folder, exist_ok=True)
    keyframes_folder = os.path.join(output_folder, "keyframes")
    os.makedirs(keyframes_folder, exist_ok=True)
    vector_folder = os.path.join(output_folder, "vector_file")
    os.makedirs(vector_folder, exist_ok=True)

    # --- New logic to get total frames for the progress bar ---
    total_frames = 0
    fps = 25 # Default fps
    try:
        with av.open(video_path, 'r') as container_check:
            stream_check = container_check.streams.video[0]
            fps = float(stream_check.average_rate)
            total_frames = stream_check.frames
            if total_frames == 0: # Fallback if metadata is missing
                total_frames = int(stream_check.duration * stream_check.time_base * fps)
    except (av.AVError, IndexError, TypeError) as e:
        print(f"[GPU {device_id}] Could not open or find video stream in {video_path}: {e}")
        return 0

    if total_frames == 0:
        print(f"[GPU {device_id}] Could not determine total frames for {video_path}. Aborting.")
        return 0
    # --- End of new logic ---

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
    model = model.to(device).eval()

    pbar = tqdm(total=total_frames, desc=f"[GPU {device_id}] Processing {os.path.basename(video_path)}")
    all_keyframes = 0

    try:
        with av.open(video_path, 'r', options={"threads": "auto"}) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            frames, frame_indices = [], []
            prev_features = None
            prev_keyframe_id = None

            for frame in container.decode(stream):
                pbar.update(1) # Update progress for every frame decoded
                current_frame_idx = int(frame.pts * stream.time_base * fps)

                # Sampling condition remains the same
                stride = max(1, int(fps / sample_rate))
                if (current_frame_idx % stride == 0) and ((current_frame_idx // stride) % (skip_frames + 1) == 0):
                    frame_rgb = frame.to_ndarray(format='rgb24')
                    frames.append(frame_rgb)
                    frame_indices.append(current_frame_idx)

                    if len(frames) == batch_size:
                        features_batch = extract_features(frames, model, preprocess, device)
                        features_batch = features_batch / features_batch.norm(dim=-1, keepdim=True).clamp_min(1e-12)

                        for i, (img, feat) in enumerate(zip(frames, features_batch)):
                            frame_id = frame_indices[i]
                            if is_keyframe(feat, prev_features, frame_id, prev_keyframe_id,
                                           frame_distance_threshold, clip_threshold,
                                           proximity_threshold, proximity_clip_threshold):
                                save_image(img, os.path.join(keyframes_folder, f"keyframe_{frame_id}.webp"))
                                np.savez_compressed(os.path.join(vector_folder, f"keyframe_{frame_id}.npz"),
                                                    feature=feat.cpu().numpy())
                                prev_features = feat
                                prev_keyframe_id = frame_id
                                all_keyframes += 1
                        frames, frame_indices = [], [] # Reset batch
                        
            if frames:  # still some unprocessed frames
                features_batch = extract_features(frames, model, preprocess, device)
                features_batch = features_batch / features_batch.norm(dim=-1, keepdim=True).clamp_min(1e-12)

                for i, (img, feat) in enumerate(zip(frames, features_batch)):
                    frame_id = frame_indices[i]
                    if is_keyframe(feat, prev_features, frame_id, prev_keyframe_id,
                                frame_distance_threshold, clip_threshold,
                                proximity_threshold, proximity_clip_threshold):
                        save_image(img, os.path.join(keyframes_folder, f"keyframe_{frame_id}.webp"))
                        np.savez_compressed(os.path.join(vector_folder, f"keyframe_{frame_id}.npz"),
                                            feature=feat.cpu().numpy())
                        prev_features = feat
                        prev_keyframe_id = frame_id
                        all_keyframes += 1

    except av.AVError as e:
        print(f"[GPU {device_id}] AVError while processing {video_path}: {str(e)}")
    
    pbar.close()

    if os.path.exists(keyframes_folder) and not os.listdir(keyframes_folder):
        print(f"No keyframes found for {video_path}. Deleting empty output folders.")
        os.rmdir(keyframes_folder)
        os.rmdir(vector_folder)
        os.rmdir(output_folder)

    return all_keyframes

def process_all_videos_worker(video_queue, output_base_folder, clip_threshold,
                              frame_distance_threshold, proximity_threshold, proximity_clip_threshold,
                              batch_size, sample_rate, skip_frames, device_id):

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    # temporary model to load ignore feats
    model_tmp, _, preprocess_tmp = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
    model_tmp = model_tmp.to(device).eval()
    
    del model_tmp
    torch.cuda.empty_cache()

    while True:
        try:
            video_path = video_queue.get(timeout=3)
        except Empty:
            break

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # --- SCENES FILE LOGIC REMOVED ---

        output_folder = os.path.join(output_base_folder, video_name)
        # if (Path(output_folder).exists()): 
        #     print(f"{output_folder} is already exists")
        #     continue

        print(f"[GPU {device_id}] Starting video: {video_name}")
        try:
            # --- UPDATED CALL to process_video ---
            number_of_keyframes = process_video(video_path, output_folder,
                                      clip_threshold, frame_distance_threshold,
                                      proximity_threshold, proximity_clip_threshold,
                                      batch_size, sample_rate, skip_frames, device_id)
            print(f"[GPU {device_id}] Finished video: {video_name}, keyframes: {number_of_keyframes}")
        except Exception as e:
            print(f"[GPU {device_id}] Error processing {video_name}: {str(e)}")
            
def check_video_process_all_frame_ids(video_name: str, input_folder: Path, output_base_folder: Path):
    """
    Checks if a video has been processed by verifying its last keyframe ID against the video's total frames.
    Uses PyAV to get video metadata.
    """
    video_path = input_folder / (video_name + ".mp4")
    keyframes_dir = output_base_folder / video_name / "keyframes"

    # 1. Check if the keyframes directory exists
    if not keyframes_dir.exists():
        return False

    # 2. Find the frame ID of the last keyframe generated
    try:
        # This is a robust way to check if the directory is empty and find the max in one pass
        keyframe_paths = list(keyframes_dir.glob("*.webp"))
        if not keyframe_paths:
            print(f"Info: Keyframe directory for {video_name} exists but is empty.")
            return False
        
        last_frame_id = max(int(p.stem.split('_')[-1]) for p in keyframe_paths)
    except (ValueError, IndexError):
        # Handles cases with malformed filenames or empty directory after glob
        print(f"Warning: Could not parse keyframe names in {keyframes_dir}. Assuming reprocessing is needed.")
        return False

    # 3. Get the total number of frames from the video file using PyAV
    total_frames = 0
    try:
        with av.open(str(video_path), 'r') as container:
            stream = container.streams.video[0]
            # stream.frames contains the total frame count from metadata
            total_frames = stream.frames
            
            # Fallback: If `stream.frames` is 0 (metadata missing), calculate from duration and framerate
            if total_frames == 0 and stream.duration and stream.average_rate:
                fps = float(stream.average_rate)
                duration_sec = float(stream.duration * stream.time_base)
                total_frames = int(duration_sec * fps)
                print(f"Warning: stream.frames was 0 for {video_name}. Calculated {total_frames} frames from duration.")

    except (av.AVError, IndexError, TypeError) as e:
        print(f"Error opening video {video_name} with PyAV to get frame count: {e}")
        # If we can't inspect the video, we can't confirm it's done, so return False to re-process.
        return False

    if total_frames == 0:
        print(f"Error: Could not determine total frames for {video_name}. Assuming reprocessing is needed.")
        return False

    # 4. Compare the last keyframe with the total frames
    # The threshold `25 * 60` represents a 60-second gap at 25 FPS.
    frame_difference = total_frames - last_frame_id
    if frame_difference > 25 * 60:
        print(f"Problem: {video_name}. Last keyframe is at {last_frame_id}, but video has {total_frames} frames. Gap: {frame_difference} frames.")
        return False

    # If all checks pass, the video is considered processed.
    return True


if __name__ == "__main__":
    mp.set_start_method('spawn')

    input_folder = '/mlcv1/Datasets/HCMAI25/full'
    shot_folder = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/shot_batch2'
    output_base_folder = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/supplement'
    
    # video_files = sorted(glob.glob(os.path.join(input_folder, '*.mp4')))
    num_gpus = torch.cuda.device_count()
    
    # clip_threshold = 0.96
    # frame_distance_threshold = 75
    # proximity_threshold = 15
    # proximity_clip_threshold = 0.80
    # batch_size = 4
    # sample_rate = 25
    # skip_frames = 5
    
    clip_threshold = 0.96
    frame_distance_threshold = 60
    proximity_threshold = None
    proximity_clip_threshold = None
    batch_size = 16
    sample_rate = 25
    skip_frames = 4

    video_queue = mp.Queue()
    
    # SET UP VIDEO INTERNAL
    # internal: [start_video, end_video)
    # output_base_folder += '_skip=' + str(skip_frames) + "_" + str(clip_threshold) + "_" + str(frame_distance_threshold)
    print("output folder: ", output_base_folder)
    # start_video = 'L22_V001' # include this video
    # end_video = 'L22_V002' # exclude this video
    # print("start video", start_video)
    # print("end video", end_video)
    # video_files = video_files[::-1]
    # print('reverse')q
    
    video_names = ['K01_V003', 'K01_V020', 'K01_V021', 'K01_V022', 'K01_V023', 'K01_V027', 'K01_V028', 'K01_V029', 'K01_V030', 'K02_V002', 'K02_V004', 'K02_V005', 'K02_V008', 'K02_V011', 'K02_V015', 'K02_V018', 'K02_V019', 'K02_V020', 'K02_V022', 'K02_V024', 'K02_V025', 'K02_V026', 'K02_V027', 'K02_V028', 'K02_V029', 'K02_V030', 'K02_V031', 'K03_V001', 'K03_V002', 'K03_V003', 'K03_V004', 'K03_V005', 'K03_V006', 'K03_V007', 'K03_V008', 'K03_V009', 'K03_V010', 'K03_V011', 'K03_V012', 'K03_V013', 'K03_V014', 'K03_V015', 'K03_V016', 'K03_V017', 'K03_V018', 'K03_V019', 'K03_V020', 'K03_V021', 'K03_V022', 'K03_V023', 'K03_V024', 'K03_V025', 'K03_V026', 'K03_V027', 'K03_V028', 'K03_V029', 'K04_V001', 'K04_V002', 'K04_V003', 'K04_V004', 'K04_V005', 'K04_V006', 'K04_V007', 'K04_V008', 'K04_V009', 'K04_V010', 'K04_V011', 'K04_V012', 'K04_V013', 'K04_V014', 'K04_V015', 'K04_V016', 'K04_V017', 'K04_V018', 'K04_V019', 'K04_V020', 'K04_V021', 'K04_V022', 'K04_V023', 'K04_V024', 'K04_V025', 'K04_V026', 'K04_V027', 'K04_V028', 'K04_V029', 'K04_V030', 'K05_V001', 'K05_V002', 'K05_V003', 'K05_V005', 'K05_V006', 'K05_V008', 'K07_V012', 'K07_V015', 'K07_V016', 'K07_V018', 'K07_V019', 'K07_V020', 'K07_V021', 'K07_V031', 'K08_V001', 'K08_V002', 'K08_V006', 'K08_V008', 'K08_V010', 'K08_V011', 'K08_V012', 'K08_V014', 'K08_V018', 'K08_V019', 'K08_V020', 'K08_V021', 'K08_V023', 'K09_V001', 'K09_V002', 'K09_V003', 'K09_V004', 'K09_V005', 'K09_V006', 'K09_V007', 'K09_V008']
    # video_names = ['K09_V012', 'K09_V013', 'K09_V014', 'K09_V022', 'K09_V023', 'K09_V027', 'K09_V028', 'K10_V002', 'K10_V003', 'K10_V004', 'K10_V005', 'K10_V006', 'K10_V008', 'K10_V010', 'K10_V014', 'K10_V015', 'K10_V016', 'K10_V019', 'K10_V022', 'K10_V023', 'K10_V024', 'K10_V025', 'K10_V027', 'K11_V001', 'K11_V002', 'K11_V005', 'K11_V007', 'K11_V010', 'K11_V016', 'K11_V017', 'K11_V018', 'K11_V024', 'K11_V026', 'K11_V027', 'K12_V001', 'K12_V002', 'K12_V003', 'K12_V004', 'K12_V005', 'K12_V006', 'K12_V007', 'K12_V008', 'K12_V010', 'K12_V016', 'K12_V017', 'K12_V018', 'K12_V019', 'K12_V020', 'K12_V025', 'K16_V005', 'L21_V001', 'L21_V002', 'L21_V003', 'L21_V005', 'L21_V006', 'L21_V007', 'L21_V008', 'L21_V009', 'L21_V010', 'L21_V011', 'L21_V012', 'L21_V013', 'L21_V014', 'L21_V015', 'L21_V016', 'L21_V017', 'L21_V018', 'L21_V019', 'L21_V021', 'L21_V022', 'L21_V023', 'L21_V024', 'L21_V025', 'L21_V026', 'L21_V027', 'L21_V028', 'L21_V029', 'L21_V030', 'L21_V031', 'L22_V001', 'L22_V002', 'L22_V003', 'L22_V004', 'L22_V005', 'L22_V006', 'L22_V007', 'L22_V008', 'L22_V009', 'L22_V010', 'L22_V011', 'L22_V012', 'L22_V013', 'L22_V014', 'L22_V015', 'L22_V016', 'L22_V017', 'L22_V018', 'L22_V019', 'L22_V020', 'L22_V021', 'L22_V022', 'L22_V023', 'L22_V024', 'L22_V025', 'L22_V026', 'L22_V027', 'L22_V028', 'L22_V029', 'L22_V030', 'L22_V031', 'L24_V004', 'L25_V007', 'L25_V010', 'L25_V027', 'L25_V051', 'L25_V054', 'L25_V056', 'L26_V037', 'L26_V411', 'L28_V004', 'L30_V030']
    # video_names = video_names[::-1]
    # print("reverse")
    print(video_names)
    
    iii = input("Checking your output folder, type [y/n]")
    if iii != 'y': 
        exit()
    
    video_files = []
    for _video_name in video_names:
        _video = Path(input_folder) / (_video_name + ".mp4")
        video_files.append(_video)
    
    for vf in tqdm(video_files):
        video_name = os.path.splitext(os.path.basename(vf))[0]
        # if (start_video <= video_name and video_name < end_video): 
        if not check_video_process_all_frame_ids(video_name=video_name, input_folder=Path(input_folder), output_base_folder=Path(output_base_folder)):
            video_queue.put(vf)
    
    processes = []
    for device_id in range(num_gpus):
        # --- ARGUMENTS LIST UPDATED ---
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