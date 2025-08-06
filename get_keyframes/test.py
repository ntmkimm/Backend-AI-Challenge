import cv2
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import open_clip
import csv
import glob
import concurrent.futures
import torch.multiprocessing as mp
from queue import Queue, Empty
import numpy as np
import threading
import time
from typing import List, Tuple, Optional
import torchvision.transforms as transforms

# Global model cache to avoid reloading
model_cache = {}
cache_lock = threading.Lock()

def get_model(device_id):
    """Get or create model for specific device with caching"""
    with cache_lock:
        if device_id not in model_cache:
            device = torch.device(f"cuda:{device_id}")
            print(f"Loading model for device: {device}...")
            torch.cuda.set_device(device_id)
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-H-14-378-quickgelu', pretrained='dfn5b'
            )
            model = model.to(device).eval()
            model = model.half()  # Enable half precision for faster inference
            
            # Compile model for faster inference (PyTorch 2.0+)
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode='max-autotune')
                
            model_cache[device_id] = (model, preprocess, device)
        return model_cache[device_id]

class VideoFrameDataset(Dataset):
    """Dataset for efficiently loading video frames in batches"""
    
    def __init__(self, video_path: str, skip_frames: int = 5, max_frames: Optional[int] = None):
        self.video_path = video_path
        self.skip_frames = skip_frames
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Pre-calculate frame indices to process
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Generate frame indices (every skip_frames+1 frames)
        self.frame_indices = list(range(0, total_frames, skip_frames + 1))
        if max_frames:
            self.frame_indices = self.frame_indices[:max_frames]
        
        # Pre-load all frames into memory for faster access (if reasonable size)
        self.frames_cache = {}
        self.preload_frames = len(self.frame_indices) < 10000  # Only preload if < 10k frames
        
        if self.preload_frames:
            self._preload_frames()
    
    def _preload_frames(self):
        """Preload frames into memory for faster access"""
        print(f"Preloading {len(self.frame_indices)} frames...")
        for idx in tqdm(self.frame_indices, desc="Preloading frames"):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if ret:
                self.frames_cache[idx] = frame
    
    def __len__(self):
        return len(self.frame_indices)
    
    def __getitem__(self, idx):
        frame_id = self.frame_indices[idx]
        
        if self.preload_frames and frame_id in self.frames_cache:
            frame = self.frames_cache[frame_id]
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = self.cap.read()
            if not ret:
                # Return a dummy frame if read fails
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        return {
            'frame': frame,
            'frame_id': frame_id,
            'timestamp': frame_id / self.fps
        }
    
    def cleanup(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        self.frames_cache.clear()

def collate_frame_batch(batch):
    """Custom collate function for frame batches"""
    frames = []
    frame_ids = []
    timestamps = []
    
    for item in batch:
        if item['frame'] is not None:
            frames.append(item['frame'])
            frame_ids.append(item['frame_id'])
            timestamps.append(item['timestamp'])
    
    return {
        'frames': frames,
        'frame_ids': frame_ids,
        'timestamps': timestamps
    }

class OptimizedPreprocessor:
    """Optimized preprocessing with tensor operations"""
    
    def __init__(self, preprocess_fn, device):
        self.device = device
        self.preprocess_fn = preprocess_fn
        
        # Create optimized transform pipeline
        self.tensor_transform = transforms.Compose([
            transforms.ToPILImage(),
            preprocess_fn
        ])
    
    def __call__(self, frames_batch):
        """Process a batch of frames efficiently"""
        if not frames_batch:
            return torch.empty(0, 3, 378, 378, device=self.device, dtype=torch.half)
        
        # Convert BGR to RGB efficiently using numpy
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_batch]
        
        # Process frames in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            processed_frames = list(executor.map(self.tensor_transform, rgb_frames))
        
        # Stack tensors and move to device
        batch_tensor = torch.stack(processed_frames).to(self.device, dtype=torch.half)
        return batch_tensor

def extract_features_batch_optimized(frames_batch, model, preprocessor, device):
    """Optimized batch feature extraction with memory management"""
    if not frames_batch:
        return torch.empty(0, model.visual.output_dim, device=device, dtype=torch.half)
    
    # Preprocess frames
    images_tensor = preprocessor(frames_batch)
    
    # Extract features with automatic mixed precision
    with torch.no_grad(), torch.cuda.amp.autocast():
        features = model.encode_image(images_tensor)
        features = torch.nn.functional.normalize(features, dim=-1)
    
    # Clean up intermediate tensors
    del images_tensor
    torch.cuda.empty_cache()
    
    return features

def is_keyframe_vectorized_optimized(curr_features, prev_features, curr_frame_ids, prev_frame_id, 
                                   frame_distance_threshold, clip_threshold, proximity_threshold, 
                                   proximity_clip_threshold):
    """Optimized vectorized keyframe detection"""
    if prev_features is None or len(curr_features) == 0:
        is_key = torch.zeros(len(curr_features), dtype=torch.bool, device=curr_features.device)
        if len(curr_features) > 0:
            is_key[0] = True
        return is_key

    # Ensure tensors are on the same device
    prev_features = prev_features.to(curr_features.device)
    
    # Compute similarities efficiently
    similarities = torch.mm(curr_features, prev_features.unsqueeze(1)).squeeze()
    if similarities.dim() == 0:  # Handle single frame case
        similarities = similarities.unsqueeze(0)
    
    frame_distances = torch.tensor(curr_frame_ids, device=curr_features.device, dtype=torch.long) - prev_frame_id
    
    # Vectorized conditions using bitwise operations for speed
    distance_cond = frame_distances >= frame_distance_threshold
    proximity_cond = (frame_distances < proximity_threshold) & (similarities < proximity_clip_threshold)
    clip_cond = (frame_distances >= proximity_threshold) & (similarities < clip_threshold)
    
    return distance_cond | proximity_cond | clip_cond

class ImageSaver:
    """Optimized async image saver with batch processing"""
    
    def __init__(self, quality=85, resize_factor=0.5, max_workers=4):
        self.queue = Queue(maxsize=1000)  # Limit queue size to prevent memory issues
        self.quality = quality
        self.resize_factor = resize_factor
        self.active = True
        self.max_workers = max_workers
        
        # Use thread pool for parallel image saving
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def _worker(self):
        """Worker thread that manages the thread pool for image saving"""
        futures = []
        while self.active:
            try:
                item = self.queue.get(timeout=1)
                if item is None:  # Sentinel for shutdown
                    break
                
                # Submit to thread pool
                future = self.executor.submit(self._save_image, item[0], item[1])
                futures.append(future)
                
                # Clean up completed futures
                futures = [f for f in futures if not f.done()]
                
                self.queue.task_done()
            except Empty:
                continue
        
        # Wait for all remaining saves to complete
        concurrent.futures.wait(futures)

    def _save_image(self, img, path):
        """Optimized image saving with WebP format"""
        try:
            if img is None or img.size == 0:
                return
            
            # Resize if needed
            if self.resize_factor != 1.0:
                height, width = img.shape[:2]
                new_size = (int(width * self.resize_factor), int(height * self.resize_factor))
                img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save with optimized WebP settings
            img_pil.save(path, format="WEBP", quality=self.quality, method=6, 
                        lossless=False, optimize=True)
        except Exception as e:
            print(f"[ImageSaver] Error saving {path}: {e}")

    def add(self, img, path):
        """Add image to save queue (non-blocking)"""
        try:
            self.queue.put((img.copy(), path), timeout=5)
        except:
            print(f"[ImageSaver] Queue full, skipping {path}")

    def shutdown(self):
        """Shutdown the image saver"""
        self.queue.put(None)
        self.worker_thread.join()
        self.executor.shutdown(wait=True)

class OptimizedVideoProcessor:
    """Optimized video processor using PyTorch Dataset and DataLoader"""
    
    def __init__(self, device_id, clip_threshold, frame_distance_threshold, 
                 proximity_threshold, proximity_clip_threshold, batch_size, 
                 skip_frames, output_base_folder, num_workers=4, **kwargs):
        self.device_id = device_id
        self.model, self.preprocess, self.device = get_model(device_id)
        self.preprocessor = OptimizedPreprocessor(self.preprocess, self.device)
        
        # Hyperparameters
        self.clip_threshold = clip_threshold
        self.frame_distance_threshold = frame_distance_threshold
        self.proximity_threshold = proximity_threshold
        self.proximity_clip_threshold = proximity_clip_threshold
        self.batch_size = batch_size
        self.skip_frames = skip_frames
        self.num_workers = num_workers
        # Accept any additional kwargs without causing errors
    
    def process_video(self, video_path, output_folder, maps_folder, image_saver):
        """Process video using Dataset and DataLoader for optimal performance"""
        keyframes_folder = os.path.join(output_folder, "keyframes")
        
        # Create dataset and dataloader
        try:
            dataset = VideoFrameDataset(video_path, skip_frames=self.skip_frames)
            dataloader = DataLoader(
                dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers,
                collate_fn=collate_frame_batch,
                pin_memory=True,
                persistent_workers=True if self.num_workers > 0 else False
            )
        except Exception as e:
            print(f"[GPU {self.device_id}] Error creating dataset for {video_path}: {e}")
            return 0
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        map_file_path = os.path.join(maps_folder, f"{video_name}_map.csv")
        
        keyframe_count = 0
        prev_features = None
        prev_keyframe_id = -self.frame_distance_threshold
        
        # Progress bar
        pbar = tqdm(total=len(dataset), desc=f"[GPU {self.device_id}] {video_name[:20]}", unit="batch")
        
        with open(map_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['FrameID', 'Timestamp_sec'])
            
            for batch in dataloader:
                if not batch['frames']:
                    continue
                
                try:
                    # Extract features for the batch
                    features = extract_features_batch_optimized(
                        batch['frames'], self.model, self.preprocessor, self.device
                    )
                    
                    # Determine keyframes
                    is_key_mask = is_keyframe_vectorized_optimized(
                        features, prev_features, batch['frame_ids'], prev_keyframe_id,
                        self.frame_distance_threshold, self.clip_threshold, 
                        self.proximity_threshold, self.proximity_clip_threshold
                    )
                    
                    # Process keyframes
                    key_indices = torch.where(is_key_mask)[0]
                    if len(key_indices) > 0:
                        for idx in key_indices:
                            idx_int = idx.item()
                            frame_id = batch['frame_ids'][idx_int]
                            timestamp = batch['timestamps'][idx_int]
                            
                            save_path = os.path.join(keyframes_folder, f"keyframe_{frame_id}.webp")
                            image_saver.add(batch['frames'][idx_int], save_path)
                            csv_writer.writerow([frame_id, f"{timestamp:.2f}"])
                        
                        # Update state
                        last_key_idx = key_indices[-1].item()
                        prev_features = features[last_key_idx].clone()
                        prev_keyframe_id = batch['frame_ids'][last_key_idx]
                        keyframe_count += len(key_indices)
                    
                except Exception as e:
                    print(f"[GPU {self.device_id}] Error processing batch: {e}")
                    torch.cuda.empty_cache()
                
                pbar.update(len(batch['frames']))
        
        pbar.close()
        dataset.cleanup()
        
        # Clean up empty folders
        if keyframe_count == 0:
            try:
                if os.path.exists(keyframes_folder):
                    os.rmdir(keyframes_folder)
                if os.path.exists(output_folder):
                    os.rmdir(output_folder)
            except OSError:
                pass
        
        return keyframe_count

def process_videos_worker_optimized(video_queue, result_queue, device_id, params):
    """Optimized worker function for each GPU process"""
    try:
        # Set CUDA device for this process
        torch.cuda.set_device(device_id)
        
        # Create a copy of params without image_save_workers for the processor
        processor_params = {k: v for k, v in params.items() if k != 'image_save_workers'}
        processor = OptimizedVideoProcessor(device_id=device_id, **processor_params)
        image_saver = ImageSaver(max_workers=params.get('image_save_workers', 4))

        while True:
            try:
                video_path = video_queue.get(timeout=5)
                if video_path is None:  # Sentinel for shutdown
                    break
                
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_folder = os.path.join(params['output_base_folder'], video_name)
                maps_folder = os.path.join(params['output_base_folder'], "maps")
                os.makedirs(maps_folder, exist_ok=True)
                
                start_time = time.time()
                num_keyframes = processor.process_video(video_path, output_folder, maps_folder, image_saver)
                processing_time = time.time() - start_time
                
                result_queue.put(f"[GPU {device_id}] ✓ {video_name}: {num_keyframes} keyframes in {processing_time:.2f}s")
                
            except Empty:
                continue
            except Exception as e:
                result_queue.put(f"[GPU {device_id}] ✗ Error on {video_path}: {e}")
        
        image_saver.shutdown()
        
    except Exception as e:
        result_queue.put(f"[GPU {device_id}] ✗ Worker initialization failed: {e}")

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # --- Enhanced Configuration ---
    input_folder = '/mlcv2/Datasets/HCMAI24/updated/videos/batch1'
    output_base_folder = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/keyframes_shot_optimized'
    
    # Optimized parameters
    params = {
        'clip_threshold': 0.96,
        'frame_distance_threshold': 75,
        'proximity_threshold': 15,
        'proximity_clip_threshold': 0.85,
        'batch_size': 16,  # Increased due to optimizations
        'skip_frames': 5,
        'output_base_folder': output_base_folder,
        'num_workers': 2,  # DataLoader workers per GPU
        'image_save_workers': 4,  # ImageSaver threads per GPU
    }
    
    video_files = sorted(glob.glob(os.path.join(input_folder, '*.mp4')))
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        print("No CUDA GPUs found. Exiting.")
        exit()
    
    print(f"🚀 Found {len(video_files)} videos and {num_gpus} GPUs.")
    print(f"📊 Batch size: {params['batch_size']}, DataLoader workers: {params['num_workers']}")

    # Pre-warm GPU models
    print("🔥 Pre-warming GPU models...")
    for device_id in range(num_gpus):
        get_model(device_id)
    
    # Setup multiprocessing
    video_queue = mp.Queue()
    result_queue = mp.Queue()

    for vf in video_files:
        video_queue.put(vf)
    for _ in range(num_gpus):
        video_queue.put(None)
    
    # Start worker processes
    processes = []
    start_time = time.time()
    
    for device_id in range(num_gpus):
        p = mp.Process(target=process_videos_worker_optimized, 
                      args=(video_queue, result_queue, device_id, params))
        p.start()
        processes.append(p)

    # Monitor progress
    completed = 0
    total_keyframes = 0
    
    while completed < len(video_files):
        try:
            result = result_queue.get(timeout=60)
            print(result)
            completed += 1
            
            # Extract keyframe count from result
            if "keyframes" in result:
                try:
                    kf_count = int(result.split(":")[1].split("keyframes")[0].strip())
                    total_keyframes += kf_count
                except:
                    pass
                    
        except Empty:
            print("⚠️  Timeout waiting for results...")
            break
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    total_time = time.time() - start_time
    print(f"\n🎉 All videos processed!")
    print(f"📈 Total: {total_keyframes} keyframes in {total_time:.2f}s")
    print(f"⚡ Average: {len(video_files)/total_time:.2f} videos/sec")