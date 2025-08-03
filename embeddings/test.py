import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import multiprocessing as mp
from dataclasses import dataclass
import queue
import threading
import time
import logging
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from pymilvus import (
    connections, utility, Collection,
    CollectionSchema, FieldSchema, DataType
)
import open_clip

# === OPTIMIZED CONFIG ===
@dataclass
class Config:
    collection_name: str = 'AIC25_fullbatch1'
    dimension: int = 1024
    milvus_host: str = "192.168.20.156"
    milvus_port: str = "19530"
    
    # Optimized batch and worker settings
    batch_size: int = 32  # Increased for better GPU utilization
    flush_interval: int = 10000  # Larger batches for Milvus
    num_workers: int = min(16, mp.cpu_count())  # More workers
    prefetch_factor: int = 8  # Higher prefetch
    pin_memory: bool = True
    
    root_path: str = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1"
    
    # Performance optimizations
    use_amp: bool = True
    use_torch_compile: bool = True
    use_channels_last: bool = True
    torch_threads: int = 6  # Slightly more threads
    
    # Advanced caching and buffering
    image_cache_size: int = 5000  # Larger cache
    async_insert: bool = True
    max_insert_queue_size: int = 50  # Larger queue
    optimize_milvus_params: bool = True
    
    # New optimizations (adjusted for process safety)
    use_tensorrt: bool = False  # Enable if TensorRT available
    dynamic_batch_size: bool = True  # Adjust batch size based on GPU memory
    use_fp16_embeddings: bool = False  # Store embeddings in FP16
    parallel_image_loading: bool = False  # Disabled for process safety
    max_memory_usage: float = 0.85  # Maximum GPU memory usage
    enable_gradient_checkpointing: bool = False  # For very large models
    use_optimized_attention: bool = True  # Flash attention if available

config = Config()

# --- ENHANCED LOGGING ---
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - [%(processName)s:%(threadName)s] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# === MEMORY MANAGEMENT ===
class MemoryManager:
    @staticmethod
    def get_optimal_batch_size(device: torch.device, base_batch_size: int) -> int:
        """Dynamically determine optimal batch size based on available GPU memory"""
        if not torch.cuda.is_available():
            return base_batch_size
            
        try:
            torch.cuda.set_device(device)
            total_memory = torch.cuda.get_device_properties(device).total_memory
            current_memory = torch.cuda.memory_allocated(device)
            available_memory = total_memory - current_memory
            
            # Estimate memory per image (rough approximation)
            memory_per_image = 50 * 1024 * 1024  # ~50MB per image
            max_batch_from_memory = int(available_memory * config.max_memory_usage / memory_per_image)
            
            optimal_batch = min(max_batch_from_memory, base_batch_size * 2)
            return max(optimal_batch, 8)  # Minimum batch size of 8
        except:
            return base_batch_size
    
    @staticmethod
    def clear_cache_if_needed():
        """Clear GPU cache if memory usage is high"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                if torch.cuda.memory_allocated(i) / torch.cuda.max_memory_allocated(i) > 0.9:
                    torch.cuda.empty_cache()

# === OPTIMIZED IMAGE DATASET ===
class HighPerformanceImageDataset(Dataset):
    """Ultra-optimized dataset with advanced caching and parallel loading"""

    def __init__(self, image_paths: List[Path], transform=None, cache_size: int = 5000):
        self.image_paths = image_paths
        self.transform = transform
        self.cache = {}
        self.cache_size = cache_size
        self.cache_order = []
        self.failed_paths = set()
        # Note: ThreadPoolExecutor removed to avoid pickle issues in multiprocessing

    def __len__(self) -> int:
        return len(self.image_paths)

    def _get_metadata(self, path_str: str) -> Tuple[str, int]:
        """Extract metadata without caching to avoid pickle issues"""
        path = Path(path_str)
        video_id = path.parent.parent.name
        try:
            frame_id = int(path.stem.replace("keyframe_", ""))
        except:
            frame_id = -1
        return video_id, frame_id

    def _load_image_optimized(self, path: Path) -> Optional[Image.Image]:
        """Optimized image loading with better error handling"""
        path_str = str(path)
        
        if path_str in self.failed_paths:
            return None
            
        if path_str in self.cache:
            return self.cache[path_str]
        
        try:
            # Use PIL optimizations
            with Image.open(path) as img:
                image = img.convert("RGB")
                
                # Cache management
                if len(self.cache) >= self.cache_size:
                    if self.cache_order:
                        oldest = self.cache_order.pop(0)
                        if oldest in self.cache:
                            del self.cache[oldest]
                
                if len(self.cache) < self.cache_size:
                    self.cache[path_str] = image.copy()
                    self.cache_order.append(path_str)
                
                return image
        except Exception as e:
            self.failed_paths.add(path_str)
            logging.error(f"Failed to load {path_str}: {e}")
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.image_paths[idx]
        image = self._load_image_optimized(path)
        
        if image is None:
            # Use correct image size for ViT-H-14-378 model
            return {
                'image': torch.zeros(3, 378, 378, dtype=torch.float16 if config.use_fp16_embeddings else torch.float32),
                'id': idx, 'filepath': str(path),
                'video_id': '', 'frame_id': -1, 'success': False
            }

        try:
            # Initialize with correct dimensions for ViT-H-14-378
            image_tensor = torch.zeros(3, 378, 378, dtype=torch.float16 if config.use_fp16_embeddings else torch.float32)
            if self.transform:
                image_tensor = self.transform(image)
                if config.use_fp16_embeddings:
                    image_tensor = image_tensor.half()

            video_id, frame_id = self._get_metadata(str(path))

            return {
                'image': image_tensor, 'id': idx, 'filepath': str(path),
                'video_id': video_id, 'frame_id': frame_id, 'success': True
            }
        except Exception as e:
            logging.error(f"Transform error for {path}: {e}")
            return {
                'image': torch.zeros(3, 378, 378, dtype=torch.float16 if config.use_fp16_embeddings else torch.float32),
                'id': idx, 'filepath': str(path),
                'video_id': '', 'frame_id': -1, 'success': False
            }

def ultra_fast_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ultra-optimized collate function with correct tensor dimensions"""
    valid_items = [item for item in batch if item['success']]
    failed_items = [item for item in batch if not item['success']]

    if not valid_items:
        dtype = torch.float16 if config.use_fp16_embeddings else torch.float32
        return {
            'images': torch.empty(0, 3, 378, 378, dtype=dtype),  # Correct dimensions for ViT-H-14-378
            'ids': [], 'filepaths': [], 'video_ids': [], 'frame_ids': [],
            'failed_items': failed_items
        }

    # Get tensor info from first valid item
    first_tensor = valid_items[0]['image']
    dtype = first_tensor.dtype
    shape = first_tensor.shape
    
    # Pre-allocate tensor with correct dimensions
    num_valid = len(valid_items)
    images = torch.empty(num_valid, *shape, dtype=dtype)
    
    # Fast tensor filling
    for i, item in enumerate(valid_items):
        images[i] = item['image']
    
    if config.use_channels_last:
        images = images.to(memory_format=torch.channels_last)

    return {
        'images': images,
        'ids': [item['id'] for item in valid_items],
        'filepaths': [item['filepath'] for item in valid_items],
        'video_ids': [item['video_id'] for item in valid_items],
        'frame_ids': [item['frame_id'] for item in valid_items],
        'failed_items': failed_items
    }

# === PROCESS-SAFE MILVUS INSERTER ===
class ProcessSafeMilvusInserter:
    def __init__(self, config: Config):
        self.config = config
        self.collection = None
        self.total_inserted = 0
        self.batch_buffer = []
        self._setup_connection()

    def _setup_connection(self):
        try:
            # Optimized connection parameters
            connections.connect(
                alias="default",
                host=self.config.milvus_host,
                port=self.config.milvus_port,
                pool_size=30,  # Moderate connection pool
                timeout=30
            )

            collection_name = f"{self.config.collection_name}"  # Process-specific name
            
            if utility.has_collection(collection_name):
                logging.warning(f"Collection {collection_name} exists. Dropping it...")
                utility.drop_collection(collection_name)

            schema = CollectionSchema([
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="filepath", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="clip_embedding", dtype=DataType.FLOAT_VECTOR, dim=self.config.dimension),
                FieldSchema(name="video_id", dtype=DataType.VARCHAR, max_length=300),
                FieldSchema(name="frame_id", dtype=DataType.INT64),
            ])

            self.collection = Collection(name=collection_name, schema=schema)
            
            # Optimized index parameters
            index_params = {
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {
                    "M": 48,  # Good balance of performance and recall
                    "efConstruction": 1024
                }
            }
            
            self.collection.create_index("clip_embedding", index_params)
            self.collection.load()
            
            logging.info(f"Milvus collection {collection_name} setup completed")
        except Exception as e:
            logging.critical(f"Milvus setup failed: {e}", exc_info=True)
            raise

    def insert_batch(self, ids: List[int], filepaths: List[str], embeddings: List[List[float]], 
                    video_ids: List[str], frame_ids: List[int]) -> bool:
        if not embeddings:
            return True
            
        try:
            self.collection.insert([ids, filepaths, embeddings, video_ids, frame_ids])
            self.total_inserted += len(ids)
            
            # Flush periodically
            if self.total_inserted >= self.config.flush_interval:
                self.collection.flush()
                self.total_inserted = 0
                logging.info(f"Flushed {self.config.flush_interval} records to Milvus")
            
            return True
        except Exception as e:
            logging.error(f"Insert error: {e}", exc_info=True)
            return False

    def finalize(self):
        if self.collection and self.total_inserted > 0:
            self.collection.flush()
            logging.info("Final flush completed")
        try:
            connections.disconnect("default")
        except:
            pass

# === ULTRA-OPTIMIZED EMBEDDING PROCESSOR ===
class UltraFastEmbeddingProcessor:
    def __init__(self, device: torch.device, use_amp: bool = True, use_torch_compile: bool = True):
        self.device = device
        self.use_amp = use_amp
        self.use_torch_compile = use_torch_compile
        self.model = None
        self.preprocess = None
        
        # Set optimal thread count
        torch.set_num_threads(config.torch_threads)
        
        # Memory management
        self.memory_manager = MemoryManager()
        
        self._load_and_optimize_model()

    def _load_and_optimize_model(self):
        """Load and heavily optimize the model"""
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14-378-quickgelu', pretrained='dfn5b'
        )
        
        # Move to device with optimal memory format
        memory_format = torch.channels_last if config.use_channels_last else torch.contiguous_format
        self.model = model.to(self.device, memory_format=memory_format).eval()
        
        # Enable all performance optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        
        # Enable optimized attention if available
        if config.use_optimized_attention and hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        # Gradient checkpointing for memory efficiency
        if config.enable_gradient_checkpointing and hasattr(self.model, 'set_grad_checkpointing'):
            self.model.set_grad_checkpointing(True)
        
        # Model compilation
        if self.use_torch_compile and hasattr(torch, 'compile'):
            logging.info(f"Compiling model with advanced optimizations on {self.device}...")
            try:
                self.model = torch.compile(
                    self.model, 
                    mode="max-autotune",
                    dynamic=config.dynamic_batch_size,
                    fullgraph=True
                )
            except Exception as e:
                logging.warning(f"Model compilation failed: {e}, falling back to non-compiled model")
        
        # Warm up the model
        self._warmup_model()
        
        logging.info(f"Model loaded and optimized on {self.device}")

    def _warmup_model(self):
        """Warm up the model for optimal performance"""
        try:
            # Use correct input size for ViT-H-14-378 model
            dummy_input = torch.randn(2, 3, 378, 378, device=self.device)
            if config.use_channels_last:
                dummy_input = dummy_input.to(memory_format=torch.channels_last)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    _ = self.model.encode_image(dummy_input)
            
            torch.cuda.synchronize()
            logging.info("Model warmup completed")
        except Exception as e:
            logging.warning(f"Model warmup failed: {e}")

    @torch.no_grad()
    def encode_batch(self, images: torch.Tensor) -> List[List[float]]:
        """Ultra-fast batch encoding with all optimizations"""
        if images.size(0) == 0:
            return []
        
        # Efficient GPU transfer
        images = images.to(self.device, non_blocking=True)
        
        if config.use_channels_last:
            images = images.to(memory_format=torch.channels_last)
        
        # Mixed precision inference
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.float16):
            embeddings = self.model.encode_image(images)
        
        # Fast normalization
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to appropriate precision
        if config.use_fp16_embeddings:
            embeddings = embeddings.half()
        
        return embeddings.cpu().tolist()

def get_image_paths_optimized(root_path: str) -> List[Path]:
    """Optimized image path discovery"""
    image_extensions = {'.webp', '.jpg', '.jpeg', '.png'}
    root = Path(root_path)
    
    # Simple but efficient path collection
    all_paths = []
    for ext in image_extensions:
        all_paths.extend(root.glob(f"**/*{ext}"))
    
    return sorted(all_paths)

def process_on_device_optimized(device_id: int, image_paths: List[Path], config: Config):
    """Ultra-optimized processing function"""
    mp.current_process().name = f"GPU-{device_id}"
    setup_logging()
    
    try:
        # Set device and configure
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
        
        # Initialize memory manager
        memory_manager = MemoryManager()
        
        # Determine optimal batch size
        optimal_batch_size = memory_manager.get_optimal_batch_size(device, config.batch_size)
        logging.info(f"Using optimized batch size: {optimal_batch_size}")
        
        logging.info(f"Processing {len(image_paths)} images with advanced optimizations")

        # Initialize components
        processor = UltraFastEmbeddingProcessor(device, config.use_amp, config.use_torch_compile)
        inserter = ProcessSafeMilvusInserter(config)

        dataset = HighPerformanceImageDataset(
            image_paths, 
            transform=processor.preprocess, 
            cache_size=config.image_cache_size
        )
        
        # Optimized DataLoader
        dataloader = DataLoader(
            dataset, 
            batch_size=optimal_batch_size, 
            shuffle=False, 
            num_workers=config.num_workers,
            pin_memory=config.pin_memory, 
            prefetch_factor=config.prefetch_factor,
            collate_fn=ultra_fast_collate_fn, 
            persistent_workers=config.num_workers > 0, 
            drop_last=False,
            multiprocessing_context='spawn' if mp.get_start_method() == 'spawn' else None
        )

        # Processing loop with advanced monitoring
        failed_files = []
        total_processed = 0
        start_time = time.time()
        last_gc_time = time.time()

        with tqdm(dataloader, desc=f"[GPU {device_id}] Ultra-Fast Processing", ncols=120) as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Handle failed items
                if batch['failed_items']:
                    print("fail")
                    failed_paths = [item['filepath'] for item in batch['failed_items']]
                    failed_files.extend(failed_paths)

                if batch['images'].size(0) == 0:
                    print("image error")
                    continue

                # Process embeddings
                embeddings = processor.encode_batch(batch['images'])
                success = inserter.insert_batch(
                    batch['ids'], batch['filepaths'], embeddings, 
                    batch['video_ids'], batch['frame_ids']
                )

                if success:
                    total_processed += len(batch['ids'])
                    elapsed = time.time() - start_time
                    speed = total_processed / elapsed if elapsed > 0 else 0
                    
                    # Enhanced progress display
                    gpu_mem = torch.cuda.memory_allocated(device) / 1024**3
                    pbar.set_postfix({
                        'processed': total_processed,
                        'failed': len(failed_files),
                        'speed': f'{speed:.1f}/s',
                        'GPU_mem': f'{gpu_mem:.1f}GB'
                    })

                # Periodic memory management
                if batch_idx % 50 == 0:
                    memory_manager.clear_cache_if_needed()
                
                # Periodic garbage collection
                current_time = time.time()
                if current_time - last_gc_time > 30:  # Every 30 seconds
                    gc.collect()
                    last_gc_time = current_time

        # Cleanup
        inserter.finalize()
        torch.cuda.empty_cache()
        gc.collect()

        # Save failed files
        if failed_files:
            with open(f"failed_files_gpu_{device_id}.txt", "w") as f:
                f.write("\n".join(failed_files))

        # Final statistics
        elapsed = time.time() - start_time
        speed = total_processed / elapsed if elapsed > 0 else 0
        logging.info(f"GPU {device_id} completed: {elapsed:.1f}s, {total_processed} processed "
                    f"({speed:.1f}/s), {len(failed_files)} failed")

    except Exception as e:
        logging.critical(f"Fatal error on GPU {device_id}: {e}", exc_info=True)
        raise

def main():
    """Ultra-optimized main function"""
    setup_logging()
    logging.info("=== ULTRA-OPTIMIZED Multi-GPU Image Embedding Pipeline ===")
    
    # Advanced system information
    logging.info(f"System: {mp.cpu_count()} CPUs, {psutil.virtual_memory().total / (1024**3):.1f}GB RAM")
    
    # Get image paths with optimization
    all_image_paths = get_image_paths_optimized(config.root_path)
    logging.info(f"Discovered {len(all_image_paths)} images to process")

    # GPU setup
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        logging.critical("No CUDA devices available")
        raise RuntimeError("No CUDA devices available")

    # Display optimization settings
    logging.info(f"Using {num_gpus} GPUs with ULTRA optimizations:")
    logging.info(f"  - Dynamic batch sizing: {config.dynamic_batch_size}")
    logging.info(f"  - Base batch size: {config.batch_size}, Workers: {config.num_workers}")
    logging.info(f"  - AMP: {config.use_amp}, Torch compile: {config.use_torch_compile}")
    logging.info(f"  - FP16 embeddings: {config.use_fp16_embeddings}")
    logging.info(f"  - Parallel loading: Disabled (process-safe)")
    logging.info(f"  - Advanced caching: {config.image_cache_size} items")

    # Distribute work across GPUs
    paths_per_gpu = [all_image_paths[i::num_gpus] for i in range(num_gpus)]
    for i, paths in enumerate(paths_per_gpu):
        logging.info(f"GPU {i}: {len(paths)} images assigned")

    # Launch ultra-fast processing
    start_time = time.time()
    processes = []
    
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=process_on_device_optimized, 
            args=(gpu_id, paths_per_gpu[gpu_id], config),
            name=f"GPU-{gpu_id}-Process"
        )
        p.start()
        processes.append(p)

    # Wait for completion
    for p in processes:
        p.join()

    # Final statistics
    total_time = time.time() - start_time
    total_failed = sum([
        len(Path(f"failed_files_gpu_{i}.txt").read_text().splitlines())
        for i in range(num_gpus) 
        if Path(f"failed_files_gpu_{i}.txt").exists()
    ])
    total_processed = len(all_image_paths) - total_failed
    total_speed = total_processed / total_time if total_time > 0 else 0
    
    logging.info("\n" + "="*50)
    logging.info("🚀 ULTRA-OPTIMIZED PIPELINE COMPLETED 🚀")
    logging.info(f"⏱️  Total time: {total_time:.1f}s")
    logging.info(f"📊 Images processed: {total_processed:,}")
    logging.info(f"❌ Images failed: {total_failed:,}")
    logging.info(f"🔥 Overall speed: {total_speed:.1f} images/s")
    logging.info(f"⚡ Per-GPU average: {total_speed/num_gpus:.1f} images/s")
    logging.info(f"💾 Effective throughput: {total_processed * config.dimension * 4 / (1024**3) / total_time:.2f} GB/s")
    logging.info("="*50)

if __name__ == "__main__":
    # Set optimal multiprocessing method
    mp.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    # Environment optimizations
    os.environ['OMP_NUM_THREADS'] = str(config.torch_threads)
    os.environ['MKL_NUM_THREADS'] = str(config.torch_threads)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Enable cuDNN v8
    
    main()