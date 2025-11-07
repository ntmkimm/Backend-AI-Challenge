from pathlib import Path
import json
import re

root_folder = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge")
surya_folder = Path('./json/fullframe_surya')

def clean_text(s):
    if not s:
        return ""

    s = s.lower()
    remove_patterns = [
    r"<[^>]+>",                     # Xóa HTML tags như <b>, </b>, <i>...
    r"[•٠]",                        # Xóa ký tự bullet '•' và '٠'
    r"-", r"\.", r"/cdot", r"<\*>", 
    r"h\s*\|\s*t\s*\|\s*v", r"h\s*t\s*v", r"h\s*t", r"htv",
    r"\b(?:giay|giãy|giây)\b",      # Xóa từ giây/giay/giãy
    r"\b\d+\s*(?:s|sec|second|seconds)\b",
    r"\b\d{1,2}\s*:\s*\d{1,2}\b",
    r"\b\d{1,2}\s*:\s*\d{1,2}\s*\d?\b",
]

    for pat in remove_patterns:
        s = re.sub(pat, "", s)

    # Xóa ký tự lặp lại liên tiếp (>=3 lần) thành 1 ký tự
    s = re.sub(r'(.)\1{2,}', r'\1', s)

    # Xóa khoảng trắng dư thừa
    s = re.sub(r"\s+", " ", s).strip()

    return s

c = 0
for video_folder in sorted(root_folder.iterdir()):
    if "L25" in video_folder.name:
        continue

    video_id = video_folder.name
    surya_file = surya_folder / f"{video_id}.json"
    parseq_file = video_folder / "ocr_parseq_newmodel.json"
    output_file = video_folder / "ocr.json"
    
    data1, data2 = {}, {}
    if not parseq_file.exists():
        print(f"Missing file parseq for {video_id}")
        continue
    else:
        with open(parseq_file, "r") as f2:
            data2 = json.load(f2)

    if not surya_file.exists() :
        print(f"Missing file surya for {video_id}")
        c += 1
    else:
        with open(surya_file, "r") as f1:
            data1 = json.load(f1)
        
    data = {}

    kf_files = sorted(
        (video_folder / "keyframes").glob("*.webp"),
        key=lambda x: int(x.stem[9:])  # sort theo số sau 'keyframe_'
    )
    
    for kf_file in kf_files:
        kf_id = kf_file.stem[9:]

        str1 = data1.get(kf_id, "")
        str2 = data2.get(kf_id, "")

        if str1 and str1[0] == "7":
            str1 = str1[1:]
        if str2 and str2[0] == "7":
            str2 = str2[1:]

        clean1 = clean_text(str1)
        clean2 = clean_text(str2)

        data[kf_id] = clean1 + " " + clean2

    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved cleaned OCR for {video_id}")

print(c)