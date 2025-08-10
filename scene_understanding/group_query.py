from pathlib import Path

root = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/pack1-groupA/pack1-groupA")

data = []

for _id, _txt in enumerate(sorted(root.glob("*.txt"))):
    with open(_txt, "r", encoding="utf-8") as f:
        text = f.read()
    data.append(str(_id) + "\n")
    data.append(text + "\n\n")  # Add file content followed by newline

with open("all_queries.txt", "w", encoding="utf-8") as fi:
    fi.writelines(data)