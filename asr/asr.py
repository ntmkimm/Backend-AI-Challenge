import csv
import subprocess
import tempfile
import json
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
import os

# Load PhoWhisper with batch support
transcriber = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-base",
    return_timestamps=False,
    batch_size=8
)

DATASET = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1")
ROOT = Path("/mlcv2/Datasets/HCMAI24/streaming/batch1_audio/")
MAP = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1/maps")

# Iterate over each video
for _vid in sorted(ROOT.iterdir()):
    video_id = _vid.stem
    m3u8_path = _vid / (video_id + "_720p.m3u8")
    csv_file = MAP / (video_id + "_map.csv")

    if not m3u8_path.exists() or not csv_file.exists():
        print(f"⏭️ Skipping {video_id}: missing files")
        continue

    # Load frame timestamps
    frame_ids, timestamps = [], []
    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_ids.append(row["Frame ID"])
            timestamps.append(float(row["Seconds"]))

    results = {}
    step = 3
    overlap_seconds = 1.0
    batch_size = 8

    segment_batches = []
    frame_id_batches = []

    print(f"⏳ Preparing segments for {video_id}...")

    for i in range(0, len(timestamps) - step, step):
        raw_start = timestamps[i]
        start = max(0.0, raw_start - overlap_seconds)
        end = timestamps[i + step] if i + step < len(timestamps) else timestamps[-1]
        frame_batch = frame_ids[i : min(i + step, len(frame_ids))]

        tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", str(m3u8_path),
                "-t", str(end - start),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                tmp_audio.name
            ], check=True)

            segment_batches.append(tmp_audio.name)
            frame_id_batches.append(frame_batch)

        except Exception as e:
            print(f"❌ ffmpeg error for {video_id} frame {frame_batch}: {e}")
            for fid in frame_batch:
                results[fid] = ""
            try:
                os.remove(tmp_audio.name)
            except Exception:
                pass
            continue

    print(f"🎙️ Transcribing {video_id} in {len(segment_batches)} chunks...")

    for i in tqdm(range(0, len(segment_batches), batch_size), desc=f"Transcribing {video_id}"):
        batch_paths = segment_batches[i : i + batch_size]
        batch_frame_groups = frame_id_batches[i : i + batch_size]

        try:
            batch_results = transcriber(batch_paths)
            if not isinstance(batch_results, list):
                batch_results = [batch_results]

            for frames, out in zip(batch_frame_groups, batch_results):
                transcript = out.get("text", "").strip()
                for fid in frames:
                    results[fid] = transcript

        except Exception as e:
            print(f"❌ Transcription error in batch {i}: {e}")
            for frames in batch_frame_groups:
                for fid in frames:
                    results[fid] = ""
        finally:
            # ✅ Clean up temp files
            for path in batch_paths:
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"⚠️ Could not delete {path}: {e}")

    # Save output
    output_path = DATASET / video_id / "asr.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Saved to: {output_path}\n")
