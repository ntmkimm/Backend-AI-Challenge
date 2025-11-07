import os
import argparse
from pathlib import Path

os.environ["PROTONX_API_KEY"] = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6ImRhdHBoYW5iY0BnbWFpbC5jb20iLCJpYXQiOjE3NjE1NDc2MDEsImV4cCI6MTc2NDEzOTYwMX0.ZMFRh8skkKrLDbFernryvu4p1GPqR86Ts95PUe_GawM"



import os
from protonx import ProtonX

client = ProtonX()

input_path = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/ocr/ProtonX/12282.txt'
output_path = '/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/ocr/ProtonX/12282_corrected.txt'

with open(input_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

corrected_lines = []

for i, line in enumerate(lines, 1):
    line = line.strip()
    if not line:
        corrected_lines.append("")
        continue

    try:
        result = client.text.correct(input=line, top_k=3)
        candidates = result["data"][0]["candidates"]
        best_candidate = max(candidates, key=lambda x: x["score"])  # nhanh hơn sorted
        corrected_text = best_candidate["output"]
    except Exception as e:
        print(f"[Lỗi dòng {i}] {e}")
        corrected_text = line

    corrected_lines.append(corrected_text)
    print(f"[{i}/{len(lines)}] ✅ {line} → {corrected_text}")

# Ghi kết quả
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(corrected_lines))

print(f"\n✅ Đã lưu kết quả hiệu chỉnh vào: {output_path}")



# correct_text = client.text.correct(input = "Toi di hoc", top_k=3)
