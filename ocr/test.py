import os
import re
import json
import av
import cv2
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types
from tqdm import tqdm
import argparse

# -------------------------------
# Gemini Client
# -------------------------------
video_data = Path("/mlcv1/Datasets/HCMAI25/full")
input_folder = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge")
output_folder = Path("./qwen")

video_names = ['K01_V003', 'K01_V020', 'K01_V021', 'K01_V022', 'K01_V023', 'K01_V027', 'K01_V028', 'K01_V029', 'K01_V030', 'K02_V002', 'K02_V004', 'K02_V005', 'K02_V008', 'K02_V011', 'K02_V015', 'K02_V018', 'K02_V019', 'K02_V020', 'K02_V022', 'K02_V024', 'K02_V025', 'K02_V026', 'K02_V027', 'K02_V028', 'K02_V029', 'K02_V030', 'K02_V031', 'K03_V001', 'K03_V002', 'K03_V003', 'K03_V004', 'K03_V005', 'K03_V006', 'K03_V007', 'K03_V008', 'K03_V009', 'K03_V010', 'K03_V011', 'K03_V012', 'K03_V013', 'K03_V014', 'K03_V015', 'K03_V016', 'K03_V017', 'K03_V018', 'K03_V019', 'K03_V020', 'K03_V021', 'K03_V022', 'K03_V023', 'K03_V024', 'K03_V025', 'K03_V026', 'K03_V027', 'K03_V028', 'K03_V029', 'K04_V001', 'K04_V002', 'K04_V003', 'K04_V004', 'K04_V005', 'K04_V006', 'K04_V007', 'K04_V008', 'K04_V009', 'K04_V010', 'K04_V011', 'K04_V012', 'K04_V013', 'K04_V014', 'K04_V015', 'K04_V016', 'K04_V017', 'K04_V018', 'K04_V019', 'K04_V020', 'K04_V021', 'K04_V022', 'K04_V023', 'K04_V024', 'K04_V025', 'K04_V026', 'K04_V027', 'K04_V028', 'K04_V029', 'K04_V030', 'K05_V001', 'K05_V002', 'K05_V003', 'K05_V005', 'K05_V006', 'K05_V008', 'K07_V012', 'K07_V015', 'K07_V016', 'K07_V018', 'K07_V019', 'K07_V020', 'K07_V021', 'K07_V031', 'K08_V001', 'K08_V002', 'K08_V006', 'K08_V008', 'K08_V010', 'K08_V011', 'K08_V012', 'K08_V014', 'K08_V018', 'K08_V019', 'K08_V020', 'K08_V021', 'K08_V023', 'K09_V001', 'K09_V002', 'K09_V003', 'K09_V004', 'K09_V005', 'K09_V006', 'K09_V007', 'K09_V008']
video_names += ['K09_V012', 'K09_V013', 'K09_V014', 'K09_V022', 'K09_V023', 'K09_V027', 'K09_V028', 'K10_V002', 'K10_V003', 'K10_V004', 'K10_V005', 'K10_V006', 'K10_V008', 'K10_V010', 'K10_V014', 'K10_V015', 'K10_V016', 'K10_V019', 'K10_V022', 'K10_V023', 'K10_V024', 'K10_V025', 'K10_V027', 'K11_V001', 'K11_V002', 'K11_V005', 'K11_V007', 'K11_V010', 'K11_V016', 'K11_V017', 'K11_V018', 'K11_V024', 'K11_V026', 'K11_V027', 'K12_V001', 'K12_V002', 'K12_V003', 'K12_V004', 'K12_V005', 'K12_V006', 'K12_V007', 'K12_V008', 'K12_V010', 'K12_V016', 'K12_V017', 'K12_V018', 'K12_V019', 'K12_V020', 'K12_V025', 'K16_V005', 'L21_V001', 'L21_V002', 'L21_V003', 'L21_V005', 'L21_V006', 'L21_V007', 'L21_V008', 'L21_V009', 'L21_V010', 'L21_V011', 'L21_V012', 'L21_V013', 'L21_V014', 'L21_V015', 'L21_V016', 'L21_V017', 'L21_V018', 'L21_V019', 'L21_V021', 'L21_V022', 'L21_V023', 'L21_V024', 'L21_V025', 'L21_V026', 'L21_V027', 'L21_V028', 'L21_V029', 'L21_V030', 'L21_V031', 'L22_V001', 'L22_V002', 'L22_V003', 'L22_V004', 'L22_V005', 'L22_V006', 'L22_V007', 'L22_V008', 'L22_V009', 'L22_V010', 'L22_V011', 'L22_V012', 'L22_V013', 'L22_V014', 'L22_V015', 'L22_V016', 'L22_V017', 'L22_V018', 'L22_V019', 'L22_V020', 'L22_V021', 'L22_V022', 'L22_V023', 'L22_V024', 'L22_V025', 'L22_V026', 'L22_V027', 'L22_V028', 'L22_V029', 'L22_V030', 'L22_V031', 'L24_V004', 'L25_V007', 'L25_V010', 'L25_V027', 'L25_V051', 'L25_V054', 'L25_V056', 'L26_V037', 'L26_V411', 'L28_V004', 'L30_V030']

numbers_need_to_remove = [
    1, 2, 3, 6, 8, 11, 17, 21, 20, 23, 24, 31, 34, 37, 39, 40, 41, 42,
    46, 48, 49, 52, 51, 53, 56, 57, 60, 63, 65, 72, 78, 79, 87, 90, 92,
    93, 95, 97, 98, 105, 110, 111, 113, 118, 119, 120, 121, 122, 124, 125,
    127, 131, 132, 133, 134, 138, 143, 144, 146, 147, 150, 153, 155, 156,
    158, 159, 161, 166, 169, 172, 174, 176, 177, 179, 183, 192, 198, 199
]

from psycopg2 import pool
import os
from dotenv import load_dotenv
import threading
from typing import Dict, List

load_dotenv()

_connection_pool = None
_pool_lock = threading.Lock()   

def get_pool():
    global _connection_pool
    if _connection_pool is None:
        with _pool_lock:
            if _connection_pool is None:  # double-checked locking
                _connection_pool = pool.SimpleConnectionPool(
                    1, 50,  
                    dbname=os.getenv("DB_NAME"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASSWORD"),
                    host=os.getenv("DB_HOST"),
                    port=os.getenv("DB_PORT")
                )
    return _connection_pool

def get_connection():
    return get_pool().getconn()

def release_connection(conn):
    get_pool().putconn(conn)


def get_label_cluster(
    video_id: str,
    frame_index: int,
) -> int:
    """User dislike cluster label của một keyframe"""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT label
        FROM cluster
        WHERE video_id = %s AND frame_id = %s
    """, (video_id, frame_index))
    
    row = cur.fetchone()
    cur.close()
    release_connection(conn)
    
    try:
        label = row[0]
    except:
        return None

    return label


parser = argparse.ArgumentParser(description="Batch OCR with Gemini on keyframes")
parser.add_argument("--START", type=str, required=True, help="Start video stem (e.g., L21_V001)")
parser.add_argument("--END", type=str, required=True, help="End video stem (e.g., L25_V001)")
parser.add_argument("--API", type=int, required=True, help="Index 0-based")
args = parser.parse_args()

apis = [
    "AIzaSyD5M1-0OT1jqVXT92uBFgWCPKzxpwAePIQ",
    "AIzaSyBnPdbks51-h8qCcu_Iith_XZkBZRTp_1A",
    "AIzaSyCPfAcUcxahDD1Es-jhJCnrqQDcsXFbWRA",
    "AIzaSyCik_mnL2BZeg-X4MKml89aCioPQ9rlr50",
    "AIzaSyCCRFZGeqj5IAGpw9OejUJ3ItkAER9JZnw",
    "AIzaSyAnY0ytQX0i1jOUw-e_CMkRU87syN3EcLI",
    "AIzaSyCqspYoG6pvOz2V2WfHzu9hxXS8O_qsO6Y",
    "AIzaSyA7mYUndJMczy5eGIMCHJZpPYrcyRzISn4",
    "AIzaSyDeNGE_kzmVm12pJn2KICB7cMUqT2QayHM",
    "AIzaSyAzS6oiRme6pgqIYDzUIJP7y5ZL_ktJbEU",
    "AIzaSyAowZ112nFRMoA_66I49FyJJBk3dzvwxaY",
    
    "AIzaSyCYx9oYUdxqopgs3vixDAra6lcnckQ-6wk",
    "AIzaSyDLKN7Y68qxhDKGiTLFHPCvg9hxGX8ZAYo",
    "AIzaSyC7xdu0MI-x_ow_9oL1R2doWCkQHApLMCM",
    "AIzaSyCMUahoYn6UJBOz269Id-__hR3lbTxc2so",
    "AIzaSyBLONlPT6QvOwWDLtT2vwI40yVH5u9CWho",
    "AIzaSyDEDyB9sG2pwOGL30fQTjpEOabGOD6XD2k",
    "AIzaSyAIzgouZNe5KNdT7tFt9XRAU5RYnwKuvXg",
    "AIzaSyB9PS73wOHItpQYXoEIEE8qa16QeaYu6A4",
    "AIzaSyBC_fcnjiNsc9rEa5sisSnqYDvHWbiIDjE",
    "AIzaSyBd-msMfpwLkV11wAswaVijhWj9rvWrrSg",
    "AIzaSyCOqlWIHZLdDtQs8F2vU0oeeMkECUBsouY",
    "AIzaSyB8DnFkLVABuJ6rQ19VdT66MTegxGhNAeE",
    "AIzaSyDxdEDsoVqjE7OarmBBy6obaoG2ZPtBJ-Q",
    "AIzaSyCElNo_jnV4uwWnvVushM1MA_Yq4VSt4SI",
    "AIzaSyAFQdYubP1cbCQ-YWTVePjLvUlKtkFUiV4",
    "AIzaSyCq6DeEYkjgdA0wG283vRrZdHTkyspKHhs",
    "AIzaSyDPyM9TOINPq8Z7enlyRLj4wYqJwJiWcCE",
    "AIzaSyDj6bDR3ygbHUZdqqndXZD31K3HuW4o6QM",
    "AIzaSyBi5eduHC07ogOvfDL3o8ThE3GhHKeDL3A",
    "AIzaSyDbHXTpzT1XEdkBq18vck_RitgB4VNvLAI",
    "AIzaSyDxAJOTMyrcHxCzECPhaQDQIGx8gTDdyjA",
    "AIzaSyA14hS1fGUTeC4Zd7xi6bzTp9EKbbzUsfg",
    "AIzaSyBYh7rAJq8nysigR7Vj5OlCDLHpvi27kSc",
    "AIzaSyBSKyVnYUVj9p_6YQpvpvQlYZT_UWN-AFE",
    "AIzaSyBhdk0Ay68qyQqSvmtejQCfHCuAgRNlaSI",
    "AIzaSyBCXuV93O7kKfNZLOSCyvGult86xnjAv6M",
    "AIzaSyCekdP4rzF-e68h2oGyY3cybp_J6lRRqig",
    
    "AIzaSyB3Tlbw2PVyx8Qjd-YZ_7NfyIctij8SSt4",
    "AIzaSyAF2-c1eBqAQZUE7s9hoadtUqmV8DiN51I",
    "AIzaSyCvg0qWKn5wMfSGx8DECrai8k_vw0yo4lA",
    "AIzaSyAL1SBZYsTmud390cbu5jrP9bLIDLrnGak",
    
    
    'AIzaSyC_Yq_bG2EpXqkdTHeYQlTngQSWauDkjU0',
    'AIzaSyAKTvkUEX4Hgl-lmZv5ICIWygbYGQIfexw',
    'AIzaSyBYARdMMRJv9mavO-PGUFBbvYYzExZiEwE',
    'AIzaSyBBqDecXerKG-MqIT47Uz7yHCtdii08-hU',
    'AIzaSyDNjFdngMjTAxsHOF8GszUMtmQb64zXSj8',
    'AIzaSyCgOcqqZnP-5l9QMtl78eiPvQkk_V88kKM',
    'AIzaSyCSdyqKPhpmlhAS0-RuO22Wo8uPtiG6njw',
    'AIzaSyCqkLtdqhBOD0KARsjzXsOidEJ2yvbyTmM',
    'AIzaSyChb72g_U-TKTJDMQqc4KloeE6v6xMi1MI',
    'AIzaSyDD8TgP2YzprmJo9ZPyjxcxikrYsewDGFo',
    'AIzaSyCllaR1171sWYYONFyTcNMQobudoq2TALQ',
    'AIzaSyAN6HuRMtgaC5-Txktnbk034fipARBR6Pc',
    'AIzaSyC8H0TD_n_Sa3pkXU3NJq3PI9P59xhEx3M',
    'AIzaSyAzpjZRSsgUbmqxkIveuKRF2mXvAkH6P50',
    'AIzaSyD6NgPickO_dScmJfFf0YLKYmJP1ZdNWT0',
    'AIzaSyBgLFtHXyAhEd7PQ4zImjW5uK-JOKGIotY',
    'AIzaSyDQBJzcDs1GoAtLYIYBHSsqLAS2Uv0Cyfw',
    'AIzaSyDXQrGHH6P4P8IXxgEHxrqU2RhVW5_aGo8',
    'AIzaSyBUKXgB6ofAr5MHd2GNAHfQjAOWuCTfGmw',
    'AIzaSyCEtKBlis8uGMk_wPT3Iz3AANoHaLTIArw',
    'AIzaSyAmvPchUoljShzuJ5k5jR-AWue_yVwbVAA',
    'AIzaSyC3b_nMNmcPLesoRXyBGrU3xZjAzKiKVu8',
    'AIzaSyC_Dl1P4BPoe9qwYh1zZ6l911no2KKB0VY',
    'AIzaSyBJIZCov1ncmIytRhnzpZLQ8gy9JoTSlSo',
    'AIzaSyDgTzRSGPggzefksL3QW6DUHUfjsJcBUR8',
    'AIzaSyDMvvyesY_Px3bR4jGlJuUOWLq70EfUpR0',
    'AIzaSyCBs351yGjLSaM5Xe6YYQrzVe1TMfhWVVM',
    'AIzaSyCNBbuU7D6ISVqiptfpS-7JL9P9VEtkMno',
    'AIzaSyCzsvHJFaK8sPobZOpzY21GCea1VQUSw9g',
    'AIzaSyBnLwi1XTsO70iLnRIlIY_6m2_ajIMucTY',
    'AIzaSyBHzp65YsKW3dZBh8gmMP9qLFvlJerCSv8',
    'AIzaSyBB_OCU-Qd2M2rp3plHK8FIbKJ0GHJalGo',
    'AIzaSyDwPQ-7Lu9g-fzIyhRfLNrLrtDhk4OxR1E',
    'AIzaSyAioM3kXm6guCM8NDtvuFed19v838ZXgwA',
    'AIzaSyBnPiA6-fd8ktmN72V99XAAQFJD3BjFE0s',
    'AIzaSyDMm_3w7bJMIJLmr2N52KJFJbfDMhv9gTE',
    'AIzaSyD9NqJkA08N5S5i0zfBOEgzlYK_900nJbg',
    'AIzaSyC_04QpGR4c6lwtbjmoGmvBFk8nqdwCuhw',
    'AIzaSyC6OhCu1mja1i6I2asKoN2a44RbTdXPNsw',
    'AIzaSyDKCtKIoScEqmYMkDt68zAD_I-arJ8vdCk',
    'AIzaSyDsCqIREg3l_Dga_2yuyo0qH9yWT9IZvFM',
    'AIzaSyAdGKNDZUVpTENv8JlBbx5KOiauzEY9e6Q',
    'AIzaSyBtewJ4xJLAXv_umBNVUsPNffxebJViOS0',
    'AIzaSyC1sBMXwdJy6rase4ePcxbZsmshK56Xclw',
    'AIzaSyCX_KsJqIuwXjmJ600XwAhoBi8Dl9vDe80',
    'AIzaSyCYoD4e3vn1zwA65Vg_p6HaiyrRU9vpeF0',
    'AIzaSyB1UoWPWUWtGkIz29c2ICHTk3M_uNDCUig',
    'AIzaSyCL2eTAvk1V-RcbJdn3CwXQmUs7v9xwYmo',
    'AIzaSyD_Y1RillMFS7n5F8eLiOnDU8_kBen1lZQ',
    'AIzaSyCDKz9ZyZvcrhynJ74mTBC9ULSGi0v-8Jc',
]

import random

args.API = random.randint(0, len(apis))

client = genai.Client(
    api_key=apis[args.API],  
)
models = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]

generate_content_config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_budget=0),
)

from tenacity import retry, wait_exponential, stop_after_attempt, RetryError
from google.genai.errors import ServerError

@retry(wait=wait_exponential(min=1, max=60), stop=stop_after_attempt(5))
def generate_content_with_retry(contents, prompt, generate_content_config):
    api_index = args.API
    for _id in range(len(apis)):
        for _model_id in range(len(models)):
            try:
                client = genai.Client(api_key=apis[api_index])
                response = client.models.generate_content(
                    model=models[_model_id],
                    contents=contents + [prompt],
                    config=generate_content_config,
                )
                if response is None:
                    raise ValueError("Empty response")
                return response
            except Exception as e:
                print(f"[WARN] API index {api_index} failed: {e}")
                api_index = (api_index + 1) % len(apis)
    raise RuntimeError("All API keys failed.")


# -------------------------------
def generate_from_image(frame_img) -> str:
    """
    frame_img
    return: ocr: str
    """

    contents = []
    is_success, buffer = cv2.imencode(".jpg", frame_img)
    if not is_success:
        raise Exception(f"Could not encode frame")
    image_bytes = buffer.tobytes()
    contents.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

    prompt = f"""
Trích xuất OCR của ảnh, chú ý cả các chữ nhỏ và mờ trên các vật thể trong ảnh. Nếu là số toán học, hãy viết dưới dạng **LaTeX**
Bắt buộc trả về JSON tiếng Việt, kể cả khi OCR là chuỗi rỗng.

```json
{{
"ocr": "text bạn trích xuất được"
}}```
"""

    response = generate_content_with_retry(contents, prompt, generate_content_config)
    if response == None or response.text == None:  # Check if response is None
        raise ValueError("Response is None. Unable to process results.")
    print(response)  # Handle the successful response
    
    json_string = response.text.strip()

    # clean markdown fences
    if json_string.startswith("```"):
        json_string = json_string.strip("`").replace("json", "", 1).strip()

    # replace python-style triple quotes
    json_string = re.sub(r'"""(.*?)"""',
                            lambda m: json.dumps(m.group(1)),
                            json_string,
                            flags=re.S)
    ocr = ""
    # Try parsing as JSON
    try:
        results = {}
        data_dict = json.loads(json_string)

        # Trường hợp Gemini trả {"ocr": ""}
        if isinstance(data_dict, dict) and "ocr" in data_dict:
            ocr = data_dict.get("ocr", "")
            
        return ocr

    except json.JSONDecodeError:
        print("⚠️ JSON parse failed. Using regex.")
        match = re.search(r'"ocr"\s*:\s*"([^"]*)"', json_string)
        if match:
            ocr = match.group(1)

    return ocr
        
    

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


video_files = []
for _video_mp4_path in sorted(video_data.glob("*.mp4")):
    if _video_mp4_path.stem >= args.START and _video_mp4_path.stem < args.END:
        video_files.append(_video_mp4_path)

for _video_mp4_path in tqdm(video_files):
    if _video_mp4_path.stem in video_names: continue
    print("process video: ", _video_mp4_path.stem)
    stem = _video_mp4_path.stem
    folder_of_video = input_folder / stem
    keyframes_folder = folder_of_video / "keyframes"
    out_folder_of_video = output_folder / stem
    out_folder_of_video.mkdir(exist_ok=True, parents=True)

    keyframe_indices = sorted(int(p.stem[9:]) for p in keyframes_folder.glob("*.webp"))
    if not keyframe_indices:
        print(f"No keyframes found for {stem}")
        continue
    keyframe_indices = [4152, 4151, 4140]
    frames = extract_keyframes(_video_mp4_path, keyframe_indices[:])

    ocrs = {}

    for frame_id, frame_img in frames.items():
        if (out_folder_of_video / (str(frame_id) + ".txt")).exists(): 
            continue
        label = get_label_cluster(video_id=_video_mp4_path.stem, frame_index=frame_id)
        if label and label in numbers_need_to_remove: continue
        
        ocr = generate_from_image(frame_img=frame_img)
            
        output_file = out_folder_of_video / (str(frame_id) + ".txt")
        print(f"{frame_id}: {ocr}")
        with open(output_file, "w", encoding="utf8") as f1:
            f1.write(ocr)

print("✅ Done.")
