import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
from decord import VideoReader, cpu
from scipy.interpolate import UnivariateSpline
from tqdm import tqdm

from ultralytics import YOLO


class TStarSearcher:
    """
    A class to perform keyframe search in a video using object detection and dynamic sampling.
    """

    def __init__(
        self,
        video_path: str,
        object_detector: YOLO,
        target_objects: List[str],
        cue_objects: List[str],
        search_nframes: int = 8,
        image_grid_shape: Tuple[int, int] = (8, 8),
        search_budget: float = 0.1,
        output_dir: Optional[str] = None,
        confidence_threshold: float = 0.5,
        object2weight: Optional[dict] = None,
    ):
        self.video_path = video_path
        self.target_objects = target_objects
        self.cue_objects = cue_objects
        self.objects = target_objects + cue_objects
        self.search_nframes = search_nframes
        self.output_dir = output_dir
        self.confidence_threshold = confidence_threshold
        self.object2weight = object2weight if object2weight else {}
        self.fps = 1  # Sampling rate: 1 frame per second

        # Video properties
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {self.video_path}")
        self.raw_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = total_frames / self.raw_fps

        # Adjust total frame number based on sampling rate
        self.total_frame_num = int(self.duration * self.fps)
        self.image_grid_shape = image_grid_shape

        # Auto-adjust grid if video is too short
        grid_rows, grid_cols = self.image_grid_shape
        num_frames_in_grid = grid_rows * grid_cols
        if self.total_frame_num < num_frames_in_grid:
            side = int(np.floor(np.sqrt(self.total_frame_num)))
            side = max(side, 1)
            if side * side > self.total_frame_num:
                side -= 1
            # Use as square as possible
            new_rows = side
            new_cols = max(1, self.total_frame_num // new_rows)
            if new_rows * new_cols > self.total_frame_num:
                new_cols = new_cols - 1
            if new_cols == 0:
                new_cols = 1
            self.image_grid_shape = (new_rows, new_cols)
            print(f"Auto-adjusted grid to: {self.image_grid_shape}")

        self.remaining_targets = target_objects.copy()
        self.search_budget = min(1000, self.total_frame_num * search_budget)

        # Initialize distributions and histories
        self.score_distribution = np.zeros(self.total_frame_num) + 1e-6
        self.non_visiting_frames = np.ones(self.total_frame_num)
        self.P = np.ones(self.total_frame_num) * self.confidence_threshold * 0.3

        self.P_history = []
        self.Score_history = []
        self.non_visiting_history = []
        self.image_grid_iters = []
        self.detect_annotot_iters = []
        self.detect_bbox_iters = []

        # Set YOLO interface (object_detector)
        self.object_detector = object_detector
        self.object_detector.set_classes(target_objects + cue_objects)
        for obj in target_objects:
            self.object2weight[obj] = 1.0
        for obj in cue_objects:
            self.object2weight[obj] = 0.5

    # --- Detection Methods ---
    def imageGridScoreFunction(
        self,
        images: List[np.ndarray],
        output_dir: Optional[str],
        image_grids: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[List[List[str]]]]:
        if not images:
            return np.array([]), []

        grid_rows, grid_cols = image_grids
        grid_height = images[0].shape[0] / grid_rows
        grid_width = images[0].shape[1] / grid_cols

        confidence_maps = []
        detected_objects_maps = []

        for image in images:
            detections = self.object_detector.predict(
                source=image,
                conf=self.confidence_threshold
            )

            confidence_map = np.zeros((grid_rows, grid_cols))
            detected_objects_map = [[] for _ in range(grid_rows * grid_cols)]

            for detection in detections:
                for bbox in detection.boxes:
                    label = int(bbox.cls)
                    confidence = float(bbox.conf)
                    object_name = self.objects[label]  # Map class id to name
                    weight = self.object2weight.get(object_name, 0.5)
                    adjusted_confidence = confidence * weight

                    x_min, y_min, x_max, y_max =  bbox.xyxy[0].cpu().numpy()
                    box_center_x = (x_min + x_max) / 2
                    box_center_y = (y_min + y_max) / 2

                    grid_x = int(box_center_x // grid_width)
                    grid_y = int(box_center_y // grid_height)
                    grid_x = min(grid_x, grid_cols - 1)
                    grid_y = min(grid_y, grid_rows - 1)

                    cell_index = grid_y * grid_cols + grid_x
                    confidence_map[grid_y, grid_x] = max(confidence_map[grid_y, grid_x], adjusted_confidence)
                    detected_objects_map[cell_index].append(object_name)

            confidence_maps.append(confidence_map)
            detected_objects_maps.append(detected_objects_map)

        return np.stack(confidence_maps), detected_objects_maps

    def read_frame_batch(self, video_path: str, frame_indices: List[int]) -> Tuple[List[int], np.ndarray]:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        frame_indices = [min(int(round(idx)), total_frames-1) for idx in frame_indices]
        return frame_indices, vr.get_batch(frame_indices).asnumpy()

    def create_image_grid(self, frames: List[np.ndarray], rows: int, cols: int) -> np.ndarray:
        if len(frames) != rows * cols:
            # Pad frames with last frame if not enough
            if len(frames) == 0:
                pad_frame = np.zeros((95, 200, 3), dtype=np.uint8)
            else:
                pad_frame = frames[-1]
            frames = frames + [pad_frame] * (rows * cols - len(frames))
        resized_frames = [cv2.resize(frame, (200, 95)) for frame in frames]
        grid_rows = [np.hstack(resized_frames[i * cols:(i + 1) * cols]) for i in range(rows)]
        return np.vstack(grid_rows)

    def score_image_grids(
        self,
        images: List[np.ndarray],
        image_grids: Tuple[int, int]
    ) -> Tuple[np.ndarray, List[List[List[str]]]]:
        return self.imageGridScoreFunction(images, self.output_dir, image_grids)

    def store_score_distribution(self):
        self.P_history.append(copy.deepcopy(self.P).tolist())
        self.Score_history.append(copy.deepcopy(self.score_distribution).tolist())
        self.non_visiting_history.append(copy.deepcopy(self.non_visiting_frames).tolist())

    def update_top_25_with_window(
        self,
        frame_confidences: List[float],
        sampled_frame_indices: List[int],
        window_size: int = 5
    ):
        top_25_threshold = np.percentile(frame_confidences, 75)
        top_25_indices = [
            frame_idx for frame_idx, confidence in zip(sampled_frame_indices, frame_confidences)
            if confidence >= top_25_threshold
        ]
        for frame_idx in top_25_indices:
            for offset in range(-window_size, window_size + 1):
                neighbor_idx = frame_idx + offset
                if 0 <= neighbor_idx < len(self.score_distribution):
                    self.score_distribution[neighbor_idx] = max(
                        self.score_distribution[neighbor_idx],
                        self.score_distribution[frame_idx] / (abs(offset) + 1)
                    )

    def spline_keyframe_distribution(
        self,
        non_visiting_frames: np.ndarray,
        score_distribution: np.ndarray,
        video_length: int
    ) -> np.ndarray:
        visited_indices = np.array([idx for idx, visited in enumerate(non_visiting_frames) if visited == 0])
        observed_scores = np.array([score_distribution[idx] for idx in visited_indices])
        if len(visited_indices) == 0:
            return np.ones(video_length) / video_length

        spline = UnivariateSpline(visited_indices, observed_scores, s=0.5)
        all_frames = np.arange(video_length)
        spline_scores = spline(all_frames)

        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        adjusted_scores = np.maximum(1 / video_length, spline_scores)
        p_distribution = sigmoid(adjusted_scores)
        p_distribution /= p_distribution.sum()
        return p_distribution

    def update_frame_distribution(
        self,
        sampled_frame_indices: List[int],
        confidence_maps: np.ndarray,
        detected_objects_maps: List[List[List[str]]]
    ) -> Tuple[List[float], List[List[str]]]:
        confidence_map = confidence_maps[0]
        detected_objects_map = detected_objects_maps[0]
        grid_rows, grid_cols = self.image_grid_shape

        frame_confidences = []
        frame_detected_objects = []
        for idx, _ in enumerate(sampled_frame_indices):
            row = idx // grid_cols
            col = idx % grid_cols
            frame_confidences.append(confidence_map[row, col])
            frame_detected_objects.append(detected_objects_map[idx])

        for frame_idx, confidence in zip(sampled_frame_indices, frame_confidences):
            self.non_visiting_frames[frame_idx] = 0
            self.score_distribution[frame_idx] = confidence

        self.update_top_25_with_window(frame_confidences, sampled_frame_indices)
        self.P = self.spline_keyframe_distribution(
            self.non_visiting_frames,
            self.score_distribution,
            len(self.score_distribution)
        )
        self.store_score_distribution()

        return frame_confidences, frame_detected_objects

    # --- Sampling Methods ---
    def sample_frames(self, num_samples: int) -> Tuple[List[int], List[np.ndarray]]:
        if num_samples > self.total_frame_num:
            num_samples = self.total_frame_num

        if not self.Score_history:
            interval = max(1, self.total_frame_num // num_samples)
            sampled_frame_secs = np.arange(0, self.total_frame_num, interval)[:num_samples]
            if len(sampled_frame_secs) < num_samples:
                sampled_frame_secs = np.append(sampled_frame_secs, self.total_frame_num - 1)
        else:
            _P = (self.P + num_samples / self.total_frame_num) * self.non_visiting_frames
            threshold = np.percentile(_P, 75)
            top_25_mask = _P >= threshold
            _P = _P * top_25_mask
            if _P.sum() == 0 or np.count_nonzero(_P) < num_samples:
                print(f"Warning: Not enough non-zero entries, adjusting probability distribution.")
                _P = (self.P + num_samples / self.total_frame_num)
            _P = self.safe_normalize(_P)
            replace_flag = False if num_samples <= np.count_nonzero(_P) else True
            sampled_frame_secs = np.random.choice(
                self.total_frame_num,
                size=num_samples,
                replace=replace_flag,
                p=_P
            )

        # Map sampled indices to frame indices in the original video
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sampled_frame_indices = [
            min(int(sec / self.total_frame_num * total_frames), total_frames - 1) for sec in sampled_frame_secs
        ]
        indices, frames = self.read_frame_batch(self.video_path, sampled_frame_indices)
        resized_frames = [cv2.resize(frame, (200 * 4, 95 * 4)) for frame in frames]
        return sampled_frame_secs.tolist(), resized_frames

    def pop_frames(self, 
                    video_path: str,
                    num_samples: int):
        # _P = self.score_distribution / self.score_distribution.sum()
        _P = self.safe_normalize(self.score_distribution)
        num_samples = min(num_samples, self.total_frame_num)
        sampled_frame_secs = np.random.choice(self.total_frame_num, size=num_samples, replace=False, p=_P)
        sampled_frame_secs.sort()
        # Map sampled indices to frame indices in the original video
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices_in_video = [
            min(int(sec / self.total_frame_num * total_frames), total_frames - 1) for sec in sampled_frame_secs
        ]
        indices, frames = self.read_frame_batch(video_path, frame_indices_in_video)
        return frames, [idx / self.fps for idx in sampled_frame_secs]

    # --- Verification Methods ---
    def verify_and_remove_target(
        self,
        frame_sec: int,
        detected_objects: List[str],
        confidence_threshold: float,
    ) -> bool:
        for target in list(self.remaining_targets):
            if target in detected_objects:
                cap = cv2.VideoCapture(self.video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_idx = min(int(frame_sec / self.total_frame_num * total_frames), total_frames - 1)
                _, frame = self.read_frame_batch(self.video_path, [frame_idx])
                resized_frame = cv2.resize(frame[0], (200 * 3, 95 * 3))
                conf_map, det_obj_map = self.score_image_grids([resized_frame], (1, 1))
                single_confidence = conf_map[0, 0, 0]
                single_detected_objects = det_obj_map[0][0]
                self.score_distribution[frame_sec] = single_confidence

                if target in single_detected_objects and single_confidence > confidence_threshold:
                    self.remaining_targets.remove(target)
                    print(f"Found target '{target}' in frame {frame_idx}, score {single_confidence:.2f}")
                    return True
        return False

    # --- Main Search Logic ---
    def search(self) -> Tuple[List[np.ndarray], List[float]]:
        K = self.search_nframes
        video_length = int(self.total_frame_num)
        progress_bar = tqdm(total=video_length, desc="Searching Iterations", unit="iter", dynamic_ncols=True)

        while self.remaining_targets and self.search_budget > 0:
            grid_rows, grid_cols = self.image_grid_shape
            num_frames_in_grid = grid_rows * grid_cols
            sampled_frame_secs, frames = self.sample_frames(num_frames_in_grid)
            self.search_budget -= num_frames_in_grid
            grid_image = self.create_image_grid(frames, grid_rows, grid_cols)
            confidence_maps, detected_objects_maps = self.score_image_grids(
                images=[grid_image],
                image_grids=self.image_grid_shape
            )
            self.image_grid_iters.append([grid_image])
            frame_confidences, frame_detected_objects = self.update_frame_distribution(
                sampled_frame_indices=sampled_frame_secs,
                confidence_maps=confidence_maps,
                detected_objects_maps=detected_objects_maps
            )
            for frame_sec, detected_objects in zip(sampled_frame_secs, frame_detected_objects):
                self.verify_and_remove_target(
                    frame_sec=frame_sec,
                    detected_objects=detected_objects,
                    confidence_threshold=self.confidence_threshold,
                )
            progress_bar.update(1)
        progress_bar.close()

        k_frames, time_stamps = self.pop_frames(video_path=self.video_path, num_samples=self.search_nframes)
        return k_frames, time_stamps

    def safe_normalize(self, arr):
        arr = np.nan_to_num(arr, nan=0.0)
        arr[arr < 0] = 0.0
        total = arr.sum()
        if total == 0:
            arr = np.ones_like(arr) / len(arr)
        else:
            arr = arr / total
        # Lại chuẩn hóa một lần nữa cho chắc chắn
        arr = np.nan_to_num(arr, nan=0.0)
        arr[arr < 0] = 0.0
        if arr.sum() == 0:
            arr = np.ones_like(arr) / len(arr)
        else:
            arr = arr / arr.sum()
        return arr
    
    def search_with_visualization(self) -> Tuple[List[np.ndarray], List[float]]:
        K = self.search_nframes
        video_length = int(self.total_frame_num)
        progress_bar = tqdm(total=video_length, desc="Searching Iterations", unit="iter", dynamic_ncols=True)

        while self.remaining_targets and self.search_budget > 0:
            grid_rows, grid_cols = self.image_grid_shape
            num_frames_in_grid = grid_rows * grid_cols
            sampled_frame_secs, frames = self.sample_frames(num_frames_in_grid)
            self.search_budget -= num_frames_in_grid

            grid_image = self.create_image_grid(frames, grid_rows, grid_cols)
            confidence_maps, detected_objects_maps = self.score_image_grids(
                images=[grid_image],
                image_grids=self.image_grid_shape
            )
            frame_confidences, frame_detected_objects = self.update_frame_distribution(
                sampled_frame_indices=sampled_frame_secs,
                confidence_maps=confidence_maps,
                detected_objects_maps=detected_objects_maps
            )
            for frame_sec, detected_objects in zip(sampled_frame_secs, frame_detected_objects):
                self.verify_and_remove_target(
                    frame_sec=frame_sec,
                    detected_objects=detected_objects,
                    confidence_threshold=self.confidence_threshold,
                )
            progress_bar.update(1)
        progress_bar.close()

        k_frames, time_stamps = self.pop_frames(video_path=self.video_path, num_samples=self.search_nframes)
        return k_frames, time_stamps

# Example usage
if __name__ == "__main__":
    video_path = "./38737402-19bd-4689-9e74-3af391b15feb.mp4"
    target_objects = ["couch"]
    cue_objects = ["TV", "chair"]
    model = YOLO("yolov8l-worldv2.pt")  # Change to your weights/model

    searcher = TStarSearcher(
        video_path=video_path,
        object_detector=model,
        target_objects=target_objects,
        cue_objects=cue_objects,
        search_nframes=8,
        image_grid_shape=(8, 8),  # Will auto-reduce if video is too short
    )
    frames, times = searcher.search()
    print("Keyframes:", len(frames))
    print("Timestamps:", times)
