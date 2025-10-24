from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from config.settings import INTERVAL_JSON_FILE
import bisect

class IntervalService:
    def __init__(self):
        print("Init interval service")
        self.data = {}
        self.soft_outside_interval = 75
        try:
            with open(INTERVAL_JSON_FILE) as fi:
                self.data = fi.loads(fi)
        except:
            self.data = {}

    def get_interval(self, video_id: str, keyframe_id: int):
        frames = self.data.get(video_id, [])
        if not frames: 
            return None, None
        # Tìm vị trí chèn để giữ list sorted
        pos = bisect.bisect_left(frames, keyframe_id)

        # Lấy phần tử trước và sau
        left = None
        if pos > 0 and frames[pos - 1] - self.soft_outside_interval >= 0:
            left = frames[pos - 1] - self.soft_outside_interval
        else:
            left = None

        right = frames[pos] + self.soft_outside_interval if pos < len(frames) else None
        return left, right 