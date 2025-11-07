import json
from pathlib import Path
import polars as pl

ROOT = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge_codetr/"
JSON_ROOT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/object_detector/json/merge")
# OUTPUT_CSV = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1/objects.csv"


coco_classes = { 0: "person", 1: "bike", 2: "car", 3: "motor", 4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic_light", 10: "fire_hydrant", 11: "sign", 12: "parking_meter", 13: "bench", 14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "bag", 25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard", 32: "ball", 33: "kite", 34: "baseball_bat", 35: "glove", 36: "skateboard", 37: "surfboard", 38: "racket", 39: "bottle", 40: "glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot_dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "pot", 59: "bed", 60: "table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "fridge", 73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy_bear", 78: "hair_drier", 79: "toothbrush" }

paths = []
objects = [[] for _ in range(80)]
video_ids = []
frame_ids = []

for _vid in sorted(JSON_ROOT.iterdir()):
    video_id = _vid.stem
    with open(_vid, "r") as f:
        dic = json.load(f)
        for frame_id, objs in dic.items():
            path = ROOT + video_id + "/keyframes/keyframe_" + frame_id + ".webp"
            cnt = [0] * 80
            for obj in objs:
                _class = obj["category_id"]
                _score = obj["score"]
                if (_score < 0.6): continue
                cnt[_class] += 1
            for i, c in enumerate(cnt):
                objects[i].append(c)
            paths.append(path)
            video_ids.append(video_id)
            frame_ids.append(int(frame_id))
            
data = {}
# data["filepath"] = paths
data["video_id"] = video_ids
data["frame_id"] = frame_ids

for _id, _class in coco_classes.items():
    data[_class] = objects[_id]
    
df = pl.DataFrame(data)

print(df)

df.write_parquet(ROOT + "objects.parquet")
                
