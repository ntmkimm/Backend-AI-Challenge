import os
import sys
import cv2
import logging
import numpy as np
from typing import List, Tuple
from grounder import TStarUniversalGrounder
from heuristic import TStarSearcher
from ultralytics import YOLO
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class TStarFramework:
    """
    Main class for performing object-based frame search and question-answering in a video.
    """

    def __init__(
        self,
        video_path: str,
        object_detector: YOLO,
        grounder: TStarUniversalGrounder,
        question: str,
        search_nframes: int = 8,
        grid_rows: int = 4,
        grid_cols: int = 4,
        output_dir: str = './output',
        confidence_threshold: float = 0.6,
        search_budget: int = 1000
    ):
        self.video_path = video_path
        self.grounder = grounder
        self.object_detector = object_detector
        self.question = question
        self.search_nframes = search_nframes
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.output_dir = os.path.join(output_dir, os.path.basename(video_path).split('.')[0], question[:-1])
        self.confidence_threshold = confidence_threshold
        self.search_budget = search_budget
        self._create_output_dir()

        self.results = {} # to store search results, e.g., grounding, frames

    def _create_output_dir(self):
        """
        Ensure that the output directory exists.
        """
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """
        Run the TStar framework to search for objects and answer questions.
        """
        target_objects, cue_objects = self.get_grounded_objects()
        video_searcher = self.initialize_videoSearcher(target_objects, cue_objects)
        all_frames, time_stamps = self.perform_search(video_searcher, visualization=True)
        answer = self.perform_qa(all_frames)
        logger.info(f"Answer: {answer}")
        
        return {
            "Grounding Objects": {'target_objects': target_objects, 'cue_objects': cue_objects},
            "Frame Timestamps": time_stamps,
            "Answer": answer
        }

    def get_grounded_objects(self) -> Tuple[List[str], List[str]]:
        """
        Use Grounder to obtain target and cue objects.
        """
        target_objects, cue_objects = self.grounder.inference_query_grounding(
            video_path=self.video_path,
            question=self.question,
        )
        self.results["Grounding Objects"] = {"target_objects": target_objects, "cue_objects":cue_objects}
        logger.info(f"Target objects: {target_objects}")
        logger.info(f"Cue objects: {cue_objects}")
        return target_objects, cue_objects



    def initialize_videoSearcher(self, target_objects: List[str], cue_objects: List[str]) -> TStarSearcher:
        """
        Initialize and configure the TStarSearcher with the given objects.
        """
        videoSearcher =  TStarSearcher(
            video_path=self.video_path,
            target_objects=target_objects,
            # cue_objects=cue_objects,
            cue_objects=[],
            search_nframes=self.search_nframes,
            image_grid_shape=(self.grid_rows, self.grid_cols),
            output_dir=self.output_dir,
            confidence_threshold=self.confidence_threshold,
            search_budget=self.search_budget,
            object_detector=self.object_detector
        )

        return videoSearcher

    def perform_search(self, video_searcher: TStarSearcher, visualization: bool = False) -> Tuple[List[np.ndarray], List[float]]:
        """
        Perform the search for relevant frames and their timestamps.
        """
        if visualization:
            all_frames, time_stamps = video_searcher.search()
            self._save_frames(all_frames, time_stamps)
        else:
            all_frames, time_stamps = video_searcher.search()
        
        logger.info(f"Found {len(all_frames)} frames, timestamps: {time_stamps}")
        return all_frames, time_stamps

    def perform_qa(self, frames: List[np.ndarray]) -> str:
        """
        Perform question answering on the retrieved frames.
        """
        return self.grounder.inference_qa(
            frames=frames,
            question=self.question,
        )

    def _save_frames(self, frames: List[np.ndarray], timestamps: List[float]):
        """
        Save the relevant frames as image files.
        """
        frame_dir = os.path.join(self.output_dir, "frames")
        if os.path.exists(frame_dir):
            shutil.rmtree(frame_dir)

        os.makedirs(frame_dir, exist_ok=True)

        for idx, (frame, timestamp) in enumerate(zip(frames, timestamps)):
            frame_path = os.path.join(frame_dir, f"frame_{idx}_at_{timestamp:.2f}s.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def run_tstar(
    video_path: str,
    question: str,
    grounder: str = "gpt-4o",
    object_detector_path: str = "yolov8l-worldv2.pt",
    search_nframes: int = 16,
    grid_rows: int = 4,
    grid_cols: int = 4,
    confidence_threshold: float = 0.6,
    search_budget: float = 0.5,
    output_dir: str = './output'
):
    """
    Execute the TStar video frame search and question-answering process.
    """
    grounder = TStarUniversalGrounder(model_name=grounder)
    object_detector = YOLO(object_detector_path)

    TStarQA = TStarFramework(
        video_path=video_path,
        grounder=grounder,
        object_detector=object_detector,
        question=question,
        search_nframes=search_nframes,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        output_dir=output_dir,
        confidence_threshold=confidence_threshold,
        search_budget=search_budget
    )

    return TStarQA.run()

if __name__ == "__main__":
    # Example call to run_tstar with the appropriate arguments.
    # video_path = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/video_shot/L01_V001_shots/L01_V001_shot_0221_022394-022509.mp4"
    video_path = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/scene_understanding/TStar/LVHaystackBench/playground/03e90bbc-7d6b-423c-84d9-b5be3eff11c5.mp4"
    # question =  "What is the color of the cabinet that appears more than two times in the video?"
    # question="there is a man hold cheese, how many part of cheese that he get?"
    question = "does the woman wear rings? which finger that she wears?"
    run_tstar(video_path, question)