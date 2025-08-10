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
import shutil
from services.MetaCLIP.src.mini_clip.factory import create_model_and_transforms, get_tokenizer


MAX_FRAMES_PER_SHOT = 3
# ---------------------
# Utility functions
# ---------------------
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


def is_keyframe(curr_features, prev_features, curr_frame_id, prev_frame_id,
                frame_distance_threshold, clip_threshold, proximity_threshold, proximity_clip_threshold):
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
        with save_lock:
            img_pil.save(path, format="WEBP", quality=quality)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            raise IOError(f"Failed to write image at {path}")
    except Exception as e:
        print(f"[save_image] Error saving image {path}: {e}")


def load_ignore_features(ignore_folder, model, preprocess, device, batch_size=64):
    """Encode all images in ignore_folder to a single (N, D) tensor of L2-normalized features on `device`."""
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        paths.extend(sorted(Path(ignore_folder).glob(ext)))
    if not paths:
        print(f"[info] No images found in ignore folder: {ignore_folder}")
        return None

    feats = []
    with torch.inference_mode():
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            imgs = []
            for p in batch_paths:
                try:
                    imgs.append(preprocess(Image.open(p).convert("RGB")))
                except Exception as e:
                    print(f"[warn] Failed to open {p}: {e}")
            if not imgs:
                continue
            batch = torch.stack(imgs, 0).to(device)
            f = model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # L2-normalize
            feats.append(f)
    if not feats:
        return None
    feats = torch.cat(feats, 0).to(device)
    print(f"[info] Loaded {feats.size(0)} ignore features from {ignore_folder}")
    return feats


def process_video(video_path, scenes_file_path, output_folder, maps_folder,
                  clip_threshold, frame_distance_threshold, proximity_threshold, proximity_clip_threshold,
                  batch_size, sample_rate, skip_frames, device_id, ignore_feats=None):

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
    # model, _, preprocess = create_model_and_transforms('ViT-H-14-quickgelu-worldwide@WorldWideCLIP', pretrained='metaclip2_worldwide')
    model = model.to(device).eval()

    shot_boundaries = read_shot_boundaries(scenes_file_path)
    total_frames_to_process = sum(end - start for start, end in shot_boundaries)

    pbar = tqdm(total=total_frames_to_process, desc=f"[GPU {device_id}] Processing {os.path.basename(video_path)}")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    map_file_path = os.path.join(maps_folder, f"{video_name}_map.csv")

    all_keyframes = []
    SIM_SKIP_THRESHOLD = 0.77  # skip threshold

    def filter_by_ignore(features_batch_normed):
        if ignore_feats is None or features_batch_normed.numel() == 0:
            return torch.ones(features_batch_normed.size(0), dtype=torch.bool, device=features_batch_normed.device)
        sims = features_batch_normed @ ignore_feats.T
        max_sim, _ = sims.max(dim=1)
        keep = max_sim <= SIM_SKIP_THRESHOLD
        return keep

    with open(map_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame ID', 'Seconds'])

        for start_frame, end_frame in shot_boundaries:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # --- add: quota for this shot ---
            shot_kept = 0
            shot_quota_reached = False
            # --------------------------------

            frames, frame_indices = [], []
            prev_features = None
            prev_keyframe_id = -frame_distance_threshold

            for frame_count in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    break

                # --- add: if shot quota reached, skip heavy work but vẫn update pbar ---
                if shot_quota_reached:
                    pbar.update(1)
                    continue
                # ----------------------------------------------------------------------

                if (frame_count % (fps // sample_rate) == 0) and (frame_count // (fps // sample_rate)) % (skip_frames + 1) == 0:
                    frames.append(frame)
                    frame_indices.append(frame_count)

                    if len(frames) == batch_size:
                        features_batch = extract_features(frames, model, preprocess, device)
                        features_batch = features_batch / features_batch.norm(dim=-1, keepdim=True).clamp_min(1e-12)

                        keep_mask = filter_by_ignore(features_batch)
                        if keep_mask.sum().item() == 0:
                            frames, frame_indices = [], []
                            pbar.update(batch_size)
                            continue

                        frames = [f for f, k in zip(frames, keep_mask.tolist()) if k]
                        frame_indices = [fi for fi, k in zip(frame_indices, keep_mask.tolist()) if k]
                        features_batch = features_batch[keep_mask]

                        # === save loop with quota ===
                        for i, (frame_in_batch, feature) in enumerate(zip(frames, features_batch)):
                            if shot_quota_reached:
                                break  # quota reached mid-batch

                            frame_id = frame_indices[i]
                            if is_keyframe(feature, prev_features, frame_id, prev_keyframe_id,
                                        frame_distance_threshold, clip_threshold,
                                        proximity_threshold, proximity_clip_threshold):
                                save_image(frame_in_batch, os.path.join(keyframes_folder, f"keyframe_{frame_id}.webp"),
                                        quality=80, resize_factor=0.5)
                                np.savez_compressed(os.path.join(vector_folder, f"keyframe_{frame_id}.npz"),
                                                    feature=feature.cpu().numpy())
                                seconds = frame_id / fps
                                csv_writer.writerow([frame_id, f"{seconds:.2f}"])
                                prev_features = feature
                                prev_keyframe_id = frame_id

                                # --- add: increment and check quota ---
                                shot_kept += 1
                                if shot_kept >= MAX_FRAMES_PER_SHOT:
                                    shot_quota_reached = True
                                    # Clear buffers to avoid further processing in this shot
                                    frames, frame_indices = [], []
                                    break
                                # -------------------------------------

                        frames, frame_indices = [], []

                pbar.update(1)

            # tail leftover for the shot
            if frames and not shot_quota_reached:
                features_batch = extract_features(frames, model, preprocess, device)
                features_batch = features_batch / features_batch.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                keep_mask = filter_by_ignore(features_batch)
                if keep_mask.sum().item() != 0:
                    frames = [f for f, k in zip(frames, keep_mask.tolist()) if k]
                    frame_indices = [fi for fi, k in zip(frame_indices, keep_mask.tolist()) if k]
                    features_batch = features_batch[keep_mask]

                    for i, (frame_in_batch, feature) in enumerate(zip(frames, features_batch)):
                        if shot_quota_reached:
                            break
                        frame_id = frame_indices[i]
                        if is_keyframe(feature, prev_features, frame_id, prev_keyframe_id,
                                    frame_distance_threshold, clip_threshold,
                                    proximity_threshold, proximity_clip_threshold):
                            save_image(frame_in_batch, os.path.join(keyframes_folder, f"keyframe_{frame_id}.webp"),
                                    quality=80, resize_factor=0.5)
                            np.savez_compressed(os.path.join(vector_folder, f"keyframe_{frame_id}.npz"),
                                                feature=feature.cpu().numpy())
                            seconds = frame_id / fps
                            csv_writer.writerow([frame_id, f"{seconds:.2f}"])
                            prev_features = feature
                            prev_keyframe_id = frame_id

                            # --- add: increment and check quota ---
                            shot_kept += 1
                            if shot_kept >= MAX_FRAMES_PER_SHOT:
                                shot_quota_reached = True
                                break
                            # -------------------------------------
    cap.release()
    pbar.close()

    if os.path.exists(keyframes_folder) and not os.listdir(keyframes_folder):
        os.rmdir(keyframes_folder)
        os.rmdir(vector_folder)
        os.rmdir(output_folder)

    return all_keyframes


def process_all_videos_worker(video_queue, shot_folder, output_base_folder, clip_threshold,
                              frame_distance_threshold, proximity_threshold, proximity_clip_threshold,
                              batch_size, sample_rate, skip_frames, device_id, ignore_folder):

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device_id)
    # temporary model to load ignore feats
    model_tmp, _, preprocess_tmp = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
    # model_tmp, _, preprocess_tmp = create_model_and_transforms('ViT-H-14-quickgelu-worldwide@WorldWideCLIP', pretrained='metaclip2_worldwide')
    model_tmp = model_tmp.to(device).eval()
    ignore_feats = load_ignore_features(ignore_folder, model_tmp, preprocess_tmp, device, batch_size=64)
    del model_tmp
    torch.cuda.empty_cache()

    while True:
        try:
            video_path = video_queue.get(timeout=3)
        except Empty:
            break

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        scenes_file_path = os.path.join(shot_folder, f"{video_name}.mp4.scenes.txt")

        if not os.path.exists(scenes_file_path):
            print(f"[GPU {device_id}] Scenes file not found for {video_name}, skipping.")
            continue

        output_folder = os.path.join(output_base_folder, video_name)
        maps_folder = os.path.join(output_base_folder, "maps")
        os.makedirs(maps_folder, exist_ok=True)

        print(f"[GPU {device_id}] Starting video: {video_name}")
        try:
            keyframes = process_video(video_path, scenes_file_path, output_folder, maps_folder,
                                      clip_threshold, frame_distance_threshold,
                                      proximity_threshold, proximity_clip_threshold,
                                      batch_size, sample_rate, skip_frames, device_id, ignore_feats)
            print(f"[GPU {device_id}] Finished video: {video_name}, keyframes: {len(keyframes)}")
        except Exception as e:
            print(f"[GPU {device_id}] Error processing {video_name}: {str(e)}")


if __name__ == "__main__":
    mp.set_start_method('spawn')

    input_folder = '/mlcv2/Datasets/HCMAI24/updated/videos/batch1'
    shot_folder = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/shot'
    output_base_folder = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/keyframes_shot_dense'
    if os.path.exists(output_base_folder):
        shutil.rmtree(output_base_folder)
    ignore_folder = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/keyframes_should_ignore'

    video_files = sorted(glob.glob(os.path.join(input_folder, '*.mp4')))
    num_gpus = torch.cuda.device_count()

    # clip_threshold = 0.8
    # frame_distance_threshold = 3000
    # proximity_threshold = 15
    # proximity_clip_threshold = 0.6
    # batch_size = 8
    # sample_rate = 25
    # skip_frames = 7
    
    clip_threshold = 0.85
    frame_distance_threshold = 25 * 12
    proximity_threshold = 15
    proximity_clip_threshold = 0.75
    batch_size = 8
    sample_rate = 25
    skip_frames = 5

    video_queue = mp.Queue()
    for vf in video_files:
        # if 'L01_V026' not in vf: continue
        video_queue.put(vf)

    processes = []
    for device_id in range(num_gpus):
        p = mp.Process(target=process_all_videos_worker, args=(
            video_queue, shot_folder, output_base_folder,
            clip_threshold, frame_distance_threshold,
            proximity_threshold, proximity_clip_threshold,
            batch_size, sample_rate, skip_frames, device_id, ignore_folder))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("All videos processed.")
