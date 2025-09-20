import os
import re
import json
import av
import cv2
import base64
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, RetryError

# -------------------------------
# Config
# -------------------------------
# Bạn cần export OPENAI_API_KEY trước khi chạy:
# export OPENAI_API_KEY="sk-xxxxx"
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

batch_size = 5   # số frame xử lý 1 lần
model_name = "gpt-4o-mini"   # có thể đổi sang gpt-4o hoặc gpt-4.1

video_data = Path("/mlcv1/Datasets/HCMAI25/full")
input_folder = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge")
output_folder = Path("./gpt")

# -------------------------------
# Hàm gọi GPT có retry
# -------------------------------
@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
def generate_content_with_retry(images, prompt):
    """
    images: list of image bytes
    prompt: text prompt (string)
    """
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": prompt}
        ]}
    ]
    for img_bytes in images:
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        messages[0]["content"].append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}
        )

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )
    return response.choices[0].message.content

# -------------------------------
# Batch OCR
# -------------------------------
def generate_batch(batch_items):
    """
    batch_items: list of (frame_idx, frame_img)
    return: dict {frame_idx: ocr_text}
    """
    images = []
    frame_indices = []

    for idx, frame_img in batch_items:
        is_success, buffer = cv2.imencode(".jpg", frame_img)
        if not is_success:
            raise Exception(f"Could not encode frame {idx}")
        images.append(buffer.tobytes())
        frame_indices.append(idx)

    prompt = f"""
Bạn nhận {len(frame_indices)} ảnh. Nhiệm vụ của bạn:
1. Trích xuất *TẤT CẢ* các kí tự của từng ảnh, chú ý cả các chữ nhỏ và mờ. Nếu là số toán học, hãy viết dưới dạng **LaTeX**.
2. **Bắt buộc** trả về JSON với đúng frame_index theo format, không thêm bớt frame nào.
3. Cấu trúc JSON phải như sau:

{{
  "results": [
{os.linesep.join([f'    {{"frame_index": {i}, "ocr": ""}}' for i in frame_indices])}
  ]
}}
"""

    try:
        json_string = generate_content_with_retry(images, prompt)
        if json_string is None:
            raise ValueError("Response is None. Unable to process results.")
        print("🔎 Raw GPT output:\n", json_string)

        json_string = json_string.strip()
        if json_string.startswith("```"):
            json_string = json_string.strip("`").replace("json", "", 1).strip()

        json_string = re.sub(r'"""(.*?)"""',
                             lambda m: json.dumps(m.group(1)),
                             json_string,
                             flags=re.S)

        results = {}
        data_dict = json.loads(json_string)

        if isinstance(data_dict, dict) and "results" in data_dict:
            for item in data_dict["results"]:
                frame = str(item.get("frame_index"))
                ocr = item.get("ocr", "")
                results[frame] = ocr
        elif isinstance(data_dict, list):
            for item in data_dict:
                frame = str(item.get("frame_index"))
                ocr = item.get("ocr", "")
                results[frame] = ocr

        # Bổ sung frame thiếu
        for i in frame_indices:
            if str(i) not in results:
                results[str(i)] = ""

        return results

    except RetryError as e:
        print(f"❌ Failed after multiple retries: {e}")
        return {str(i): "" for i in frame_indices}
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return {str(i): "" for i in frame_indices}

# -------------------------------
# Extract keyframes với PyAV
# -------------------------------
def extract_keyframes(video_path, keyframe_indices):
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = {}
    keyframe_set = set(keyframe_indices)
    max_index = max(keyframe_indices) if keyframe_indices else -1

    for i, frame in enumerate(container.decode(stream)):
        if i in keyframe_set:
            frames[i] = frame.to_ndarray(format="bgr24")
        if i > max_index:
            break
    container.close()
    return frames

# -------------------------------
# Main
# -------------------------------
def main(start_stem, end_stem):
    video_files = []
    for _video_mp4_path in sorted(video_data.glob("*.mp4")):
        if _video_mp4_path.stem >= start_stem and _video_mp4_path.stem <= end_stem:
            video_files.append(_video_mp4_path)

    for _video_mp4_path in tqdm(video_files):
        print("📺 process video:", _video_mp4_path.stem)
        stem = _video_mp4_path.stem
        folder_of_video = input_folder / stem
        keyframes_folder = folder_of_video / "keyframes"
        out_folder_of_video = output_folder / stem
        out_folder_of_video.mkdir(exist_ok=True, parents=True)

        keyframe_indices = sorted(int(p.stem[9:]) for p in keyframes_folder.glob("*.webp"))
        if not keyframe_indices:
            print(f"No keyframes found for {stem}")
            continue
        keyframe_indices = [4074]
        frames = extract_keyframes(_video_mp4_path, keyframe_indices[:10])
        frame_items = list(frames.items())

        for i in tqdm(range(0, len(frame_items), batch_size), desc=f"process batch"):
            batch = frame_items[i:i+batch_size]
            output_file = out_folder_of_video / (str(batch[0][0]) + ".txt")
            if output_file.exists():
                continue
            
            results = []
            while len(results) != len(batch):
                print(1)
                results = generate_batch(batch)
                
            for _id, (f_idx, ocr) in enumerate(results.items()):
                f_idx = batch[_id][0]
                output_file = out_folder_of_video / (str(f_idx) + ".txt")
                with open(output_file, "w", encoding="utf8") as f1:
                    f1.write(ocr.strip())

    print("✅ Done.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Batch OCR with GPT on keyframes")
    parser.add_argument("--START", type=str, required=True, help="Start video stem (e.g., L21_V001)")
    parser.add_argument("--END", type=str, required=True, help="End video stem (e.g., L25_V001)")
    args = parser.parse_args()
    main(args.START, args.END)
