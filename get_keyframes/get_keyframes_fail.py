from pathlib import Path
import os
import cv2
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
import open_clip

ROOT_VIDEOS = Path("/mlcv2/Datasets/HCMAI24/updated/videos/batch1")
KEYFRAMES_ROOT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1")
ROOT_MAPS = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1/maps")

BATCH_SIZE = 64              # adjust per GPU mem
IMAGE_QUALITY = 80
RESIZE_FACTOR = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

def ensure_keyframe_saved(cap: cv2.VideoCapture, frame_id: int, image_path: Path):
    """
    Ensure the keyframe image exists at image_path.
    If missing, seek in the video and save as .webp (RGB PIL save).
    """
    if image_path.exists():
        return True

    # Seek to frame and read
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ret, frame_bgr = cap.read()
    if not ret or frame_bgr is None:
        print(f"[warn] Cannot read frame {frame_id} from video; skipping.")
        return False

    # OpenCV gives BGR -> convert to RGB for PIL
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Optional downscale
    if RESIZE_FACTOR != 1.0:
        h, w = frame_rgb.shape[:2]
        new_w = max(1, int(w * RESIZE_FACTOR))
        new_h = max(1, int(h * RESIZE_FACTOR))
        frame_rgb = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    img_pil = Image.fromarray(frame_rgb)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        img_pil.save(image_path, format="WEBP", quality=IMAGE_QUALITY)
    except Exception as e:
        print(f"[save_image] Error saving image {image_path}: {e}")
        return False

    if not image_path.exists() or os.path.getsize(image_path) == 0:
        print(f"[save_image] Failed to write image at {image_path}")
        return False

    return True


def encode_images_to_features(model, preprocess, image_paths, device=DEVICE, dtype=DTYPE):
    """
    Preprocess a list of image paths and return a (N, D) numpy array of L2-normalized features.
    """
    tensors = []
    valid_idxs = []

    for idx, p in enumerate(image_paths):
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(preprocess(img))
            valid_idxs.append(idx)
        except Exception as e:
            print(f"[warn] Failed to open/preprocess {p}: {e}")

    if not tensors:
        return None, []

    batch = torch.stack(tensors, dim=0).to(device=device, dtype=dtype)
    with torch.inference_mode():
        feats = model.encode_image(batch)
        # L2 normalize
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        feats = feats.float().cpu().numpy()

    return feats, valid_idxs


def main(batch_size=BATCH_SIZE):
    # Load CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14-378-quickgelu', pretrained='dfn5b'
    )
    model = model.to(DEVICE)
    if DEVICE == "cuda":
        model = model.to(dtype=DTYPE)
    model.eval()

    # Iterate videos
    video_paths = sorted(ROOT_VIDEOS.glob("*.mp4"))
    for video_path in tqdm(video_paths, desc="Videos"):
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[error] Cannot open video: {video_path}")
            continue

        # Map (CSV) with frames
        video_map = ROOT_MAPS / f"{video_path.stem}_map.csv"
        if not video_map.exists():
            print(f"[warn] Missing map CSV: {video_map}")
            cap.release()
            continue

        mapping = pd.read_csv(video_map)

        # Prepare folders
        kf_folder = KEYFRAMES_ROOT / video_path.stem / "keyframes"
        kf_folder.mkdir(parents=True, exist_ok=True)
        vector_folder = KEYFRAMES_ROOT / video_path.stem / "vector_file"
        vector_folder.mkdir(parents=True, exist_ok=True)

        # Collect the list of frames to process
        frame_ids_raw = mapping['Frame ID'] if 'Frame ID' in mapping.columns else mapping.iloc[:, 0]
        # Safely cast to int (CSV might store as str)
        frame_ids = []
        for f in frame_ids_raw:
            try:
                frame_ids.append(int(f))
            except:
                pass

        # First pass: ensure all required keyframe images exist (and skip unreadable frames)
        image_paths = []
        kept_frame_ids = []
        for fid in tqdm(frame_ids, leave=False, desc=f"{video_path.stem} ensure keyframes"):
            image_path = kf_folder / f"keyframe_{fid}.webp"
            ok = ensure_keyframe_saved(cap, fid, image_path)
            if ok:
                image_paths.append(image_path)
                kept_frame_ids.append(fid)

        cap.release()

        # Second pass: only encode those without vectors yet
        paths_to_encode = []
        fids_to_encode = []
        for fid, p in zip(kept_frame_ids, image_paths):
            vector_path = vector_folder / f"keyframe_{fid}.npz"
            if not vector_path.exists():  # skip if already encoded
                paths_to_encode.append(p)
                fids_to_encode.append(fid)

        if not paths_to_encode:
            # Nothing to do for this video
            continue

        # Batched encoding
        for start in tqdm(range(0, len(paths_to_encode), batch_size),
                          leave=False, desc=f"{video_path.stem} encode"):
            end = min(start + batch_size, len(paths_to_encode))
            batch_paths = paths_to_encode[start:end]
            batch_fids = fids_to_encode[start:end]

            feats, valid_idxs = encode_images_to_features(model, preprocess, batch_paths)
            if feats is None:
                continue

            # Save each feature vector
            for idx_in_batch, vec in zip(valid_idxs, feats):
                fid = batch_fids[idx_in_batch]
                vector_path = vector_folder / f"keyframe_{fid}.npz"
                try:
                    # Compressed NPZ with key 'feature'
                    np.savez_compressed(vector_path, feature=vec.astype(np.float32))
                except Exception as e:
                    print(f"[warn] Failed saving vector for frame {fid} at {vector_path}: {e}")


if __name__ == "__main__":
    main()
