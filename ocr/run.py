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
    parser.add_argument("--root-videos", default='/mlcv2/Datasets/HCMAI24/updated/videos/batch1')
    parser.add_argument("--map-folder", default='/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1/maps')
    parser.add_argument(
        "--output",
        default='/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/ocr/json/full_batch1',
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
        default=["MODEL.WEIGHTS", "models/tt_vitaev2-s_finetune_synth-tt-mlt-13-15-textocr.pth"],
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
    
    for _video_path in tqdm.tqdm(sorted(root_videos.glob("*.mp4"))):
        _video_dic = {}
        cap = cv2.VideoCapture(str(_video_path))
        _video_map = map_folder / (_video_path.stem + "_map.csv")
        mapping = pd.read_csv(_video_map)
        for frame_id in tqdm.tqdm(mapping['Frame ID']):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            bboxes, scores = demo.run_on_image(frame_bgr)
            _video_dic[frame_id] = []
            for bbox, score in zip(bboxes, scores):
                _video_dic[frame_id].append({
                    'bbox': bbox, 'score': score
                })
        cap.release()
        with open(output / (_video_path.stem + ".json"), 'w') as f:
            json.dump(_video_dic, f, indent=4)
    print("Done!!!")


# from multiprocessing import Pool, cpu_count

# def process_video(cfg, map_folder, output, video_path):
#     demo = VisualizationDemo(cfg)
#     video_path = Path(video_path)
#     video_dic = {}
#     cap = cv2.VideoCapture(str(video_path))
#     video_map = map_folder / (video_path.stem + "_map.csv")
#     mapping = pd.read_csv(video_map)
#     for frame_id in mapping['Frame ID']:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
#         ret, frame = cap.read()
#         if not ret:
#             continue
#         frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#         bboxes, scores = demo.run_on_image(frame_bgr)
#         video_dic[frame_id] = []
#         for bbox, score in zip(bboxes, scores):
#             video_dic[frame_id].append({'bbox': bbox, 'score': score})
#     cap.release()
#     with open(output / (video_path.stem + ".json"), 'w') as f:
#         json.dump(video_dic, f, indent=4)
#     print(f"Finished: {video_path}")
#     return str(video_path)

# from functools import partial
# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     args = get_parser().parse_args()
#     logger = setup_logger()
#     logger.info("Arguments: " + str(args))

#     cfg = setup_cfg(args)
#     root_videos = Path(args.root_videos)
#     map_folder = Path(args.map_folder)
#     output = Path(args.output)
#     output.mkdir(parents=True, exist_ok=True)
#     video_list = sorted(root_videos.glob("*.mp4"))

#     func = partial(process_video, cfg, map_folder, output)
#     with Pool(2) as pool:
#         for _ in tqdm.tqdm(pool.imap_unordered(func, video_list), total=len(video_list)):
#             pass
#     print("All done.")