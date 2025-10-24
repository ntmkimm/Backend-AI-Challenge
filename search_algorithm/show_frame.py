# 2 11 15 18 23 47 50 70 90 104 105 109 123 147 174 175 

# if meet L25 L26 - ignore constraint
# consider - no need (-.-): 138 152 173


from pathlib import Path 
import shutil
import json
from tqdm import tqdm 

full_folder = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge")
group_folder = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/group/merge_new_200")

# group_200 = group_folder / "group_200"
# group_200.mkdir(exist_ok=True, parents=True)

# check_groups = [68, 96, 151]
# for gr in check_groups:
#     gr_folder = group_folder / f"group_{gr}"
#     for kf_path in gr_folder.glob("*.webp"):
#         kf_name = kf_path.stem 
#         if "L25" not in kf_name:
#             print(kf_name)
#             dst_path = group_200 / kf_path.name
#             shutil.move(src=kf_path, dst=dst_path)


GAP_SHOT = 500
check_groups = [2, 11, 15, 18, 23, 47, 50, 70, 90, 104, 105, 109, 123, 147, 174, 175]
dic = {}
for gr in tqdm(check_groups):
    gr_folder = group_folder / f"group_{gr}"
    for kf_path in gr_folder.glob("*.webp"):
        kf_name = kf_path.stem 
        video_id = kf_name[:8]
        if dic.get(video_id, "") == "": dic[video_id] = []
        frame_id = int(kf_name[18:])
        dic[video_id].append(frame_id)
        
for video_id in dic.keys():
    dic[video_id] = sorted(dic[video_id])

with open("./unprocessed_interval.json", "w") as fi:
    json.dump(dic, fi, indent=2)

interval = {}
for video_id, kfs in dic.items():
    list_interval = [0]
    for _id in range(0, len(kfs)):
        if _id == 0 and kfs[0] - 0 >= GAP_SHOT: 
            list_interval.append(kfs[_id])
        elif kfs[_id] - kfs[_id - 1] >= GAP_SHOT:
            list_interval.append(kfs[_id])
            
    if len(list_interval) <= 2: continue
    interval[video_id] = list_interval

with open("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/interval.json", "w") as f:
    json.dump(interval, f, indent=2)


        



