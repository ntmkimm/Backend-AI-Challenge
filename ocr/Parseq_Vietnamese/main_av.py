import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
import json
from unidecode import unidecode
from pathlib import Path
import tqdm
import av  # Import thư viện PyAV

class Args:
    checkpoint = 'new-parseq.ckpt'
    root_videos = Path('/mlcv2/Datasets/HCMAI25/batch2/video')
    root_bboxes = Path('/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/ocr/json/batch2_2025')
    output = Path('/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/batch2')
    device = 'cuda'
    batch_size = 4
    num_workers = 1
    rotation = 0

args = Args()

# --- Phần tải model và helper functions không thay đổi ---
# Load model and image transforms
model = load_from_checkpoint(args.checkpoint).eval().to(args.device)
hp = model.hparams
datamodule = SceneTextDataModule(
    root_dir='_unused_', train_dir='_unused_', img_size=hp.img_size,
    max_label_length=hp.max_label_length, charset_train=hp.charset_train,
    charset_test=hp.charset_test, batch_size=args.batch_size,
    num_workers=args.num_workers, augment=False, rotation=args.rotation,
    remove_whitespace=False, normalize_unicode=True, min_image_dim=0,
    collate_fn=None
)
img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

# BBox helper functions (giữ nguyên)
def is_box_inside(box_big, box_small, margin=0):
    return (box_small[0] >= box_big[0] - margin and
            box_small[1] >= box_big[1] - margin and
            box_small[2] <= box_big[2] + margin and
            box_small[3] <= box_big[3] + margin)

def remove_containing_boxes(bboxes, margin=0):
    keep = []
    n = len(bboxes)
    removed_idx = set()
    for i in range(n):
        box_i = bboxes[i]['bbox']
        for j in range(n):
            if i == j: continue
            box_j = bboxes[j]['bbox']
            if is_box_inside(box_i, box_j, margin):
                removed_idx.add(i)
                break
    for i in range(n):
        if i not in removed_idx:
            keep.append(bboxes[i])
    return keep

def sort_bboxes_linewise(bboxes, y_thresh=15):
    for i, info in enumerate(bboxes):
        info['_idx'] = i
    bboxes = sorted(bboxes, key=lambda b: (b['bbox'][1], b['bbox'][0]))
    lines = []
    curr_line = []
    prev_y = None
    for info in bboxes:
        x1, y1, x2, y2 = info['bbox']
        if prev_y is None or abs(y1 - prev_y) <= y_thresh:
            curr_line.append(info)
            prev_y = y1
        else:
            lines.append(curr_line)
            curr_line = [info]
            prev_y = y1
    if curr_line:
        lines.append(curr_line)
    final_order = []
    for line in lines:
        line = sorted(line, key=lambda b: b['bbox'][0])
        final_order.extend(line)
    return final_order

def remove_bboxes_in_y_range(infos, lo_bound, up_bound):
    return [info for info in infos if not (lo_bound <= info['bbox'][1] <= up_bound and
                                           lo_bound <= info['bbox'][3] <= up_bound)]

# --- Video Range Config ---
start_video = 'K01_V001' # include this video
end_video = 'K05_V001' # not include this video
    
start_video = 'K05_V001' # include this video
end_video = 'K10_V001' # not include this video

start_video = 'K10_V001' # include this video
end_video = 'K15_V001' # not include this video

start_video = 'K15_V001' # include this video
end_video = 'K21_V001' # not include this video

print("start_video: ", start_video)
print("end_video: ", end_video)

import time
video_files = []
for _video_path in sorted(args.root_videos.glob("*.mp4")):
    video_name = _video_path.stem
    if not (start_video <= video_name < end_video):
        continue
    video_files.append(_video_path)

# --- Process Videos (Phần được viết lại) ---
for _video_path in tqdm.tqdm(video_files, desc="Overall Progress"):
    while True:
        _video_id = _video_path.stem
        print(f"\nProcessing video: {_video_id}")
        output_file = args.output / _video_id / "ocr_parseq.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if output_file.exists():
            break

        bbox_file = args.root_bboxes / (_video_id + ".json")
        if not bbox_file.exists():
            print(f"🤷 BBox file not found for {_video_id}. Skipping.")
            continue
        try:
            with open(bbox_file, 'r') as fi:
                _video_dic = json.load(fi)
        except:
            continue

        # Lấy danh sách frame ID cần xử lý và sắp xếp
        sorted_frame_ids = sorted([int(k) for k in _video_dic.keys()])
        if not sorted_frame_ids:
            print(f"🤷 No frames to process in BBox file for {_video_id}. Skipping.")
            # continue
            break
        
        _video_res = {}
        
        try:
            with av.open(str(_video_path), 'r') as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                fps = stream.average_rate
                if not fps or fps <= 0:
                    print(f"❌ Invalid FPS for {_video_id}. Skipping.")
                    # continue
                    break

                frame_id_iterator = iter(sorted_frame_ids)
                target_frame_id = next(frame_id_iterator, None)

                # Tạo progress bar cho việc duyệt video
                pbar = tqdm.tqdm(container.decode(stream), total=stream.frames, desc=f"🎞️  Scanning {_video_id}")
                
                for frame in pbar:
                    if target_frame_id is None:
                        break # Đã xử lý hết các frame cần thiết

                    current_frame_idx = int(frame.pts * frame.time_base * float(fps))
                    
                    # Bỏ qua các frame không cần thiết một cách hiệu quả
                    if current_frame_idx < target_frame_id:
                        continue
                    
                    # Khi tìm thấy frame mục tiêu
                    if current_frame_idx == target_frame_id:
                        _frame_id_str = str(target_frame_id)
                        _infos = _video_dic[_frame_id_str]

                        # Chuyển frame sang ndarray định dạng RGB
                        frame_rgb = frame.to_ndarray(format='rgb24')
                        h, w, _ = frame_rgb.shape

                        # --- Logic xử lý bbox và OCR (giữ nguyên từ code gốc) ---
                        _infos = remove_bboxes_in_y_range(_infos, lo_bound=655, up_bound=690)
                        _infos = remove_containing_boxes(_infos, margin=5)
                        _infos = sort_bboxes_linewise(_infos, y_thresh=15)
                        
                        text = ""
                        crop_imgs = []
                        _id_batch = []
                        
                        for _id_info, _info in enumerate(_infos):
                            x1, y1, x2, y2 = map(int, _info['bbox'])
                            x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
                            y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
                            if x1 >= x2 or y1 >= y2: continue

                            crop_img_arr = frame_rgb[y1:y2, x1:x2]
                            if crop_img_arr.size == 0: continue
                            crop_imgs.append(Image.fromarray(crop_img_arr))
                            _id_batch.append(_id_info)

                            if len(crop_imgs) == args.batch_size:
                                with torch.no_grad():
                                    batch = torch.stack([img_transform(img) for img in crop_imgs]).to(args.device)
                                    logits = model(batch)
                                    pred = logits.softmax(-1).detach()
                                    labels, _ = model.tokenizer.decode(pred)
                                for _idd, _label in zip(_id_batch, labels):
                                    _infos[_idd]["text"] = _label
                                text += " " + " ".join([l.lower() for l in labels])
                                crop_imgs, _id_batch = [], []
                                del batch, logits, pred
                                torch.cuda.empty_cache()

                        if crop_imgs:
                            with torch.no_grad():
                                batch = torch.stack([img_transform(img) for img in crop_imgs]).to(args.device)
                                logits = model(batch)
                                pred = logits.softmax(-1).detach()
                                labels, _ = model.tokenizer.decode(pred)
                            for _idd, _label in zip(_id_batch, labels):
                                _infos[_idd]["text"] = _label
                            text += " " + " ".join([l.lower() for l in labels])
                            del batch, logits, pred
                            torch.cuda.empty_cache()

                        _video_dic[_frame_id_str] = _infos
                        _video_res[_frame_id_str] = text.lower()
                        
                        # Lấy frame mục tiêu tiếp theo
                        target_frame_id = next(frame_id_iterator, None)
        
        except av.AVError as e:
            print(f"❌ PyAV Error while processing {_video_id}: {e}")
            # continue
            break
        except Exception as e:
            print(f"❌ An unexpected error occurred with {_video_id}: {e}")
            # continue
            break

        # --- Lưu kết quả (giữ nguyên) ---
        out_dir = args.root_bboxes.parent / (args.root_bboxes.name + "_parseq")
        out_dir.mkdir(exist_ok=True, parents=True)
        
        # Sắp xếp lại dict trước khi lưu để đảm bảo thứ tự
        _video_dic_sorted = dict(sorted(_video_dic.items(), key=lambda item: int(item[0])))
        _video_res_sorted = dict(sorted(_video_res.items(), key=lambda item: int(item[0])))

        with open(out_dir / (_video_id + ".json"), "w", encoding='utf-8') as fi:
            json.dump(_video_dic_sorted, fi, indent=4, ensure_ascii=False)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(_video_res_sorted, f, indent=4, ensure_ascii=False)

print("✅ All videos processed!")