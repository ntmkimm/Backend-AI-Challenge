# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
from pathlib import Path
import pandas as pd
import json
import av
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from utils import VisualizationDemo
from adet.config import get_cfg

# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    # cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    # cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/ocr/DeepSolo/configs/ViTAEv2_S/TotalText/finetune_150k_tt_mlt_13_15_textocr.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--root-videos", default='/mlcv1/Datasets/HCMAI25/full')
    parser.add_argument("--map-folder", default='/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge')
    parser.add_argument(
        "--output",
        default='/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/ocr/json/supplement_newmodel',
        help="A file or directory to save output json files."
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.001,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=["MODEL.WEIGHTS", "models/vitaev2-s_pretrain_synth-tt-mlt-13-15-textocr.pth"],
        nargs=argparse.REMAINDER,
    )
    
    
    return parser
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    root_videos = Path(args.root_videos)
    map_folder = Path(args.map_folder)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    
    # --> internal = [start_video, end_video)
    
    
    # start_video = 'L28_V016' # include this video
    # end_video = 'L28_V017' # not include this video
    
    start_video = 'K01_V001' # include this video
    end_video = 'K21_V001' # not include this video
    
    # start_video = 'L21_V001' # include this video
    # end_video = 'L31_V001' # not include this video
    
    print("start_video: ", start_video)
    print("end_video: ", end_video)

    video_files = []
    print("🔍 Filtering video files...")
    for _video_path in sorted(root_videos.glob("*.mp4")):
        video_name = _video_path.stem
        if start_video <= video_name and video_name < end_video:
            video_files.append(_video_path)
    
    video_files = video_files[::-1]

    print(f"✅ Found {len(video_files)} videos to process.")

    # 2. Xử lý từng video
    for _video_path in tqdm.tqdm(video_files, desc="Overall Progress"):
        video_stem = _video_path.stem
        output_path = output / (video_stem + ".json")

        if output_path.exists():
            print(f"⏩ Skipping {video_stem}, output already exists.")
            continue

        # Lấy danh sách frame ID từ thư mục keyframes tương ứng
        # LƯU Ý: Dòng mã gốc "_video_path.glob" có thể không đúng. 
        # Bạn cần đảm bảo đường dẫn đến keyframes là chính xác. 
        # Ví dụ: keyframes_dir = Path("/path/to/dataset") / video_stem / "keyframes"
        # Giả sử keyframes_dir được định nghĩa đúng ở đây.
        keyframes_dir = map_folder / video_stem / "keyframes" # <-- THAY ĐỔI ĐƯỜNG DẪN NÀY
        if not keyframes_dir.exists():
            print(f"⚠️ No keyframes directory found for {video_stem}. Skipping.")
            continue
            
        frame_ids = [int(_keyframe.stem.split('_')[-1]) for _keyframe in keyframes_dir.glob("*.webp")]
        
        if not frame_ids:
            print(f"🤷 No keyframes found for {video_stem}. Skipping.")
            continue
        # frame_ids = [4074]
        _video_dic = {}
        # Sử dụng set để tra cứu frame ID cần tìm với hiệu suất O(1)
        frame_ids_to_find = set(frame_ids)

        try:
            # Mở video bằng av.open trong một khối 'with' để quản lý tài nguyên
            with av.open(str(_video_path), 'r') as container:
                stream = container.streams.video[0]
                # Tối ưu hóa việc decode trên nhiều luồng CPU
                stream.thread_type = "AUTO" 
                
                fps = stream.average_rate
                if not fps or fps <= 0:
                    print(f"❌ Invalid FPS ({fps}) for {video_stem}. Skipping.")
                    continue

                # Tạo thanh tiến trình cho video hiện tại
                pbar = tqdm.tqdm(
                    container.decode(stream), 
                    total=stream.frames, 
                    desc=f"🎞️ Processing {video_stem}"
                )

                # Lặp qua từng frame trong video chỉ MỘT LẦN
                for frame in pbar:
                    # Nếu đã tìm thấy tất cả keyframe, thoát sớm để tiết kiệm thời gian
                    if not frame_ids_to_find:
                        break
                    
                    # Tính toán frame index từ timestamp (pts)
                    current_frame_idx = int(frame.pts * frame.time_base * float(fps))
                    
                    # Nếu frame này là một trong những frame ta cần
                    if current_frame_idx in frame_ids_to_find:
                        # Chuyển đổi frame sang mảng NumPy với định dạng BGR
                        frame_bgr = frame.to_ndarray(format='bgr24')
                        
                        # Chạy mô hình của bạn
                        bboxes, scores = demo.run_on_image(frame_bgr)
                        
                        _video_dic[current_frame_idx] = []
                        for bbox, score in zip(bboxes, scores):
                            # Đảm bảo dữ liệu có thể được serialize sang JSON
                            _video_dic[current_frame_idx].append({
                                'bbox': [int(c) for c in bbox], 
                                'score': float(score)
                            })
                        
                        # Xóa frame ID đã tìm thấy khỏi set
                        frame_ids_to_find.remove(current_frame_idx)
                
                pbar.close()

        except av.AVError as e:
            print(f"❌ Error processing video {video_stem} with PyAV: {e}")
            continue
        except Exception as e:
            print(f"❌ An unexpected error occurred with {video_stem}: {e}")
            continue

        # Kiểm tra nếu có frame nào không tìm thấy (do lỗi tính toán hoặc video bị cắt ngắn)
        if frame_ids_to_find:
            print(f"⚠️ Could not find {len(frame_ids_to_find)} frames in {video_stem}: {sorted(list(frame_ids_to_find))}")

        # Sắp xếp lại dictionary theo frame_id trước khi lưu (tùy chọn nhưng nên làm)
        sorted_video_dic = dict(sorted(_video_dic.items()))

        # Lưu kết quả
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sorted_video_dic, f, indent=4)

    print("✅ Done!!!")