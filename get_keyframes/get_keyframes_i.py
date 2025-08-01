import cv2
import os
import torch
from PIL import Image
from tqdm import tqdm
import open_clip # type: ignore
import csv
import glob
import concurrent.futures
import time


def preprocess_frame(frame, preprocess):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return preprocess(pil_image).unsqueeze(0)

def extract_features(frames, model, preprocess, device):
    # Preprocess frame images
    with concurrent.futures.ThreadPoolExecutor() as executor:
        processed_frames = list(executor.map(lambda frame: preprocess_frame(frame, preprocess), frames))

    images = torch.cat(processed_frames).to(device)  # [B, 3, H, W]

    with torch.no_grad():
        # Tự động phân phối batch lên nhiều GPU (nếu có)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        features = model(images)

    return features


def is_keyframe(curr_features, prev_features, curr_frame_id, prev_frame_id, frame_distance_threshold, clip_threshold, proximity_threshold, proximity_clip_threshold):
    # Calculate clip similarity on the GPU
    clip_similarity = torch.sum(curr_features * prev_features) / (torch.norm(curr_features) * torch.norm(prev_features))
    frame_distance = curr_frame_id - prev_frame_id
    
    if frame_distance >= frame_distance_threshold:
        return True
    elif frame_distance < proximity_threshold:
        return clip_similarity < proximity_clip_threshold
    else:
        return clip_similarity < clip_threshold

def save_image(img, path, quality=80, resize_factor=0.5):
    img_resized = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
    img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    img_pil.save(path, format="WEBP", quality=quality)

def process_video(video_path, output_folder, maps_folder, clip_threshold, frame_distance_threshold, proximity_threshold, proximity_clip_threshold, batch_size=64, sample_rate=25, skip_frames=3):
    os.makedirs(output_folder, exist_ok=True)
    keyframes_folder = os.path.join(output_folder, "keyframes")
    os.makedirs(keyframes_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
    model = model.to(device)
    model.eval()
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    
    frames = []
    frame_indices = []
    keyframes = []
    prev_features = None
    prev_keyframe_id = -frame_distance_threshold
    
    pbar = tqdm(total=total_frames, desc="Processing video")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    map_file_path = os.path.join(maps_folder, f"{video_name}_map.csv")
    
    with open(map_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Frame ID', 'Seconds'])
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every Nth frame based on sample_rate and skip_frames
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
        
        # Process any remaining frames
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
    
    return keyframes

def process_all_videos(input_folder, output_base_folder, clip_threshold, frame_distance_threshold, proximity_threshold, proximity_clip_threshold, batch_size=64, sample_rate=25,skip_frames=3):
    # Specific videos to process
    target_videos = ['L02_V014', 'L01_V006']
    
    maps_folder = os.path.join(output_base_folder, "maps")
    os.makedirs(maps_folder, exist_ok=True)
    
    for video_name in target_videos:
        video_path = os.path.join(input_folder, f"{video_name}.mp4")
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
            
        output_folder = os.path.join(output_base_folder, video_name)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"Processing video: {video_name}")
        try:
            keyframes = process_video(video_path, output_folder, maps_folder, clip_threshold, frame_distance_threshold, proximity_threshold, proximity_clip_threshold, batch_size, sample_rate,skip_frames)
            print(f"Total keyframes detected for {video_name}: {len(keyframes)}")
        except Exception as e:
            print(f"Error processing video {video_name}: {str(e)}")
        print("--------------------")


if __name__ == "__main__":
    print("Script started!")
    input_folder = '/mlcv2/Datasets/HCMAI24/updated/videos/batch1'
    output_base_folder = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/batch1'
    
    process_all_videos(input_folder, output_base_folder, 
                       clip_threshold=0.96, 
                       frame_distance_threshold=75, 
                       proximity_threshold=15,
                       proximity_clip_threshold=0.80,
                       batch_size=32,  # Tăng batch size lên
                       sample_rate=25,  # Sample 25 frames per second
                       skip_frames=5
                       )  