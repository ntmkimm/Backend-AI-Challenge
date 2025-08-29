from pathlib import Path
from tqdm import tqdm
import av  # Import thư viện PyAV

# --- Cấu hình ---
ROOT_VIDEOS = Path("/mlcv2/Datasets/HCMAI25/batch2/video")
MAP_FOLDER = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/batch2")
OUTPUT_TXT_PATH = Path("videos_with_keyframe_not_near_end.txt")
FRAME_DISTANCE_THRESHOLD = 25 * 60  # Ngưỡng: khoảng 60 giây ở 25 FPS

def is_video_problematic(video_path: Path, keyframes_base_folder: Path, threshold: int) -> bool:
    """
    Kiểm tra xem keyframe cuối cùng của video có quá xa so với cuối video hay không.

    Args:
        video_path: Đường dẫn đến tệp video.
        keyframes_base_folder: Thư mục gốc chứa các thư mục keyframe.
        threshold: Ngưỡng số khung hình được coi là "quá xa".

    Returns:
        True nếu video có vấn đề, ngược lại False.
    """
    video_name = video_path.stem
    keyframes_dir = keyframes_base_folder / video_name / "keyframes"

    if not keyframes_dir.is_dir():
        # Bỏ qua nếu không có thư mục keyframes
        return False

    # Tối ưu hóa: Lấy danh sách, sắp xếp, và chỉ phân tích cú pháp tệp cuối cùng
    keyframe_paths = list(keyframes_dir.glob("*.webp"))
    
    if not keyframe_paths:
        # Bỏ qua nếu có thư mục nhưng không có keyframe nào
        return False
    
    last_frame_id = 0
    for _keyframe in keyframe_paths:
        last_frame_id = max(last_frame_id, int(_keyframe.stem[9:]))

    # Lấy tổng số khung hình bằng PyAV một cách hiệu quả
    try:
        with av.open(str(video_path), 'r') as container:
            if not container.streams.video:
                print(f"⚠️  Không tìm thấy video stream trong file: {video_name}")
                return False
            
            stream = container.streams.video[0]
            total_frames = stream.frames
            
            # Logic dự phòng nếu metadata `frames` không tồn tại
            if total_frames == 0 and stream.duration and stream.average_rate:
                total_frames = int(stream.duration * stream.time_base * stream.average_rate)

            if total_frames == 0:
                print(f"⚠️  Không thể xác định tổng số khung hình cho: {video_name}")
                return False
            
    except av.AVError as e:
        print(f"❌ Lỗi PyAV khi xử lý video {video_name}: {e}")
        return False # Coi như không phải là video có vấn đề nếu không thể đọc được
    
    # Thực hiện kiểm tra cuối cùng
    distance = total_frames - last_frame_id
    if distance > threshold:
        print(f"PROBLEM: {video_name}, Khoảng cách: {distance} frames (Total: {total_frames}, Last KF: {last_frame_id})")
        return True
    
    return False

def main():
    """
    Hàm chính để chạy script.
    """
    problem_videos = []
    
    # Lọc các video cần kiểm tra
    videos_to_check = []
    for _video_path in sorted(ROOT_VIDEOS.glob("*.mp4")):
        video_name = _video_path.stem
        videos_to_check.append(_video_path)

    # Xử lý các video đã lọc
    print(f"🔍 Bắt đầu kiểm tra {len(videos_to_check)} video...")
    for video_path in tqdm(videos_to_check, desc="Kiểm tra videos"):
        if is_video_problematic(video_path, MAP_FOLDER, FRAME_DISTANCE_THRESHOLD):
            problem_videos.append(video_path.stem)

    # Ghi kết quả ra tệp tin
    if problem_videos:
        print(f"\n📝 Ghi {len(problem_videos)} video có vấn đề vào tệp: {OUTPUT_TXT_PATH}")
        with open(OUTPUT_TXT_PATH, "w") as f:
            for video_name in sorted(problem_videos):
                f.write(video_name + "\n")
    
    print(f"\n✅ Hoàn thành. Tìm thấy {len(problem_videos)} video có keyframe cuối cùng ở xa cuối video.")

if __name__ == "__main__":
    main()