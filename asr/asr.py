import csv
import subprocess
import tempfile
import json
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
import os
from typing import List, Optional

# ---------- Config ----------
DATASET = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/supplement")
ROOT = Path("/mlcv1/Datasets/HCMAI25/full")

# Segmenting / batching params
STEP = 3 + 2                  # number of keyframes per audio chunk
OVERLAP_SECONDS = 1.0      # left overlap to avoid cutting words
ASR_BATCH_SIZE = 16        # PhoWhisper pipeline batch size
FFMPEG_AUDIO_RATE = 16000  # Hz

# ---------- Helpers ----------
def get_fps(video_path: Path) -> Optional[float]:
    """
    Return float FPS using ffprobe r_frame_rate (e.g., '30000/1001').
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=nokey=1:noprint_wrappers=1",
                str(video_path)
            ],
            capture_output=True, text=True, check=True
        )
        rate = result.stdout.strip()  # e.g. "30000/1001" or "25/1" or "30"
        if not rate:
            return None
        if "/" in rate:
            num, den = rate.split("/")
            num = float(num)
            den = float(den)
            if den == 0:
                return None
            return num / den
        # already a number
        return float(rate)
    except Exception as e:
        print(f"⚠️ ffprobe failed for {video_path.name}: {e}")
        return None


def parse_frame_ids(video_id_dir: Path) -> List[int]:
    """
    From files like keyframe_0000123.webp, return [123, ...] sorted ascending.
    """
    frame_ids = []
    for p in sorted(video_id_dir.glob("*.webp")):
        # Assuming pattern keyframe_XXXXXX.webp; your previous code used [9:]
        name = p.stem  # "keyframe_0000123"
        # Robust parse: split by '_' and take the last token
        token = name.split("_")[-1]
        try:
            frame_ids.append(int(token))
        except ValueError:
            # skip non-standard filenames
            continue
    return sorted(frame_ids)


def build_timestamps(frame_ids: List[int], fps: float) -> List[float]:
    """
    Convert frame indices to seconds.
    """
    return [fid / fps for fid in frame_ids]


# ---------- PhoWhisper pipeline ----------
transcriber = pipeline(
    "automatic-speech-recognition",
    model="vinai/PhoWhisper-base",
    return_timestamps=False,
    batch_size=ASR_BATCH_SIZE
)

# ---------- Main ----------
# start_video = 'K01_V001' # include this video
# end_video = 'K10_V001' # not include this video

start_video = 'K01_V001' # include this video
end_video = 'L31_V001' # not include this video

print("start_video: ", start_video)
print("end_video: ", end_video)

video_files = []
for _video_path in sorted(ROOT.glob("*.mp4")):
    video_name = _video_path.stem
    if not (start_video <= video_name < end_video):
        continue
    video_files.append(_video_path)

video_files = video_files[::-1]
print("reverse")

for _vid in tqdm(video_files):
    video_id = _vid.stem
    video_dir = DATASET / video_id / "keyframes"
    output_path = DATASET / video_id / "asr.json"
    
    if (output_path.exists()):
        with open(output_path, "r") as fi:
            data = json.load(fi)
        f = False
        for k, v in data.items():
            if v == "": 
                print(f"key: {k}, value: {v}")
                f = True
                print(f"{video_id} contains empty string")
                break
        if f == False:
            print(video_id, " is already have asr file")
            continue

    # Collect keyframe frame IDs
    frame_ids = parse_frame_ids(video_dir)
    if not frame_ids:
        print(f"⚠️ No keyframe webp files found for {video_id}. Skipping.")
        continue

    # Get FPS
    fps = get_fps(_vid)
    if not fps or fps <= 0:
        print(f"❌ Could not determine FPS for {video_id}. Skipping.")
        continue

    timestamps = build_timestamps(frame_ids, fps)
    if len(timestamps) < 2:
        print(f"⚠️ Not enough timestamps for {video_id}. Skipping.")
        continue

    results = {}
    segment_batches: List[str] = []
    frame_id_batches: List[List[str]] = []

    print(f"⏳ Preparing segments for {video_id}...")

    # Build segments: each chunk spans STEP keyframes (or to the end)
    # Use the MP4 file (_vid) as the input to ffmpeg.
    # We’ll export each segment to a temp WAV at 16kHz mono.
    N = len(timestamps)
    for i in range(0, N, STEP):
        # Frames in this chunk:
        frames_chunk = frame_ids[i : min(i + STEP, N)]
        if not frames_chunk:
            continue

        # Start time with overlap (but not < 0)
        raw_start = timestamps[i]
        start = max(0.0, raw_start - OVERLAP_SECONDS)

        # End time is either the time at the next block start or the last ts.
        # We want the chunk to cover this group; pick:
        # - If we have more frames ahead, end at timestamps[min(i+STEP, N-1)]
        # - Else end at timestamps[N-1]
        end_idx = min(i + STEP, N) - 1
        end = timestamps[end_idx]

        # Ensure positive duration (if equal, pad a bit)
        duration = max(0.05, end - start)

        # Temp WAV
        tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_audio_path = tmp_audio.name
        tmp_audio.close()  # close handle so ffmpeg can write to it

        try:
            # Extract audio segment
            # Using -ss before -i is fast but less accurate; acceptable for ASR
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", str(_vid),             # <--- use the mp4 file
                    "-t", str(duration),
                    "-vn",
                    "-acodec", "pcm_s16le",
                    "-ar", str(FFMPEG_AUDIO_RATE),
                    "-ac", "1",
                    tmp_audio_path
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            segment_batches.append(tmp_audio_path)
            # Keep string frame IDs in results (consistent with your JSON keys)
            frame_id_batches.append([str(fid) for fid in frames_chunk])

        except Exception as e:
            print(f"❌ ffmpeg error for {video_id} frames {frames_chunk}: {e}")
            for fid in frames_chunk:
                results[str(fid)] = ""
            # Try cleanup of the temp file if created
            try:
                if os.path.exists(tmp_audio_path):
                    os.remove(tmp_audio_path)
            except Exception:
                pass
            continue

    print(f"🎙️ Transcribing {video_id} in {len(segment_batches)} chunks...")

    for i in tqdm(range(0, len(segment_batches), ASR_BATCH_SIZE), desc=f"Transcribing {video_id}"):
        batch_paths = segment_batches[i : i + ASR_BATCH_SIZE]
        batch_frame_groups = frame_id_batches[i : i + ASR_BATCH_SIZE]

        try:
            batch_results = transcriber(batch_paths)
            # transformers can return dict or list-of-dicts
            if isinstance(batch_results, dict):
                batch_results = [batch_results]
            if not isinstance(batch_results, list):
                # unexpected output
                raise RuntimeError(f"Unexpected ASR output type: {type(batch_results)}")

            # Guard: lengths should match; if not, truncate/pad
            n_out = min(len(batch_results), len(batch_frame_groups))
            for frames, out in zip(batch_frame_groups[:n_out], batch_results[:n_out]):
                transcript = (out.get("text") or "").strip()
                for fid in frames:
                    results[fid] = transcript

            # If outputs fewer than inputs, mark remaining as empty
            if len(batch_frame_groups) > n_out:
                for frames in batch_frame_groups[n_out:]:
                    for fid in frames:
                        results[fid] = ""

        except Exception as e:
            print(f"❌ Transcription error in batch {i}: {e}")
            for frames in batch_frame_groups:
                for fid in frames:
                    results[fid] = ""
        finally:
            # Clean up temp files for this batch
            for path in batch_paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    print(f"⚠️ Could not delete {path}: {e}")

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Saved to: {output_path}\n")
