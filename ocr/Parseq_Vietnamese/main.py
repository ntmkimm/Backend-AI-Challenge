
import torch
from PIL import Image
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint
import json
from unidecode import unidecode
from pathlib import Path
import tqdm
import cv2

class Args:
    checkpoint = 'new-parseq.ckpt'
    root_videos = Path('/mlcv2/Datasets/HCMAI24/updated/videos/batch1')
    root_bboxes = Path('/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/ocr/json/full_batch1')
    output = Path('/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full_batch1')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 1
    num_workers = 4
    rotation = 0

args = Args()

# Load model and image transforms
model = load_from_checkpoint(args.checkpoint).eval().to(args.device)
hp = model.hparams
datamodule = SceneTextDataModule(
    root_dir='_unused_',
    train_dir='_unused_',
    img_size=hp.img_size,
    max_label_length=hp.max_label_length,
    charset_train=hp.charset_train,
    charset_test=hp.charset_test,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    augment=False,
    rotation=args.rotation,
    remove_whitespace=False,
    normalize_unicode=True,
    min_image_dim=0,
    collate_fn=None
)

img_transform = SceneTextDataModule.get_transform(model.hparams.img_size)

def is_box_inside(box_big, box_small, margin=0):
    """
    Kiểm tra box_small nằm hoàn toàn trong box_big (có thể thêm margin nếu cần)
    box = [x1, y1, x2, y2]
    """
    return (box_small[0] >= box_big[0] - margin and
            box_small[1] >= box_big[1] - margin and
            box_small[2] <= box_big[2] + margin and
            box_small[3] <= box_big[3] + margin)

def remove_containing_boxes(bboxes, margin=0):
    """
    bboxes: list of dict, mỗi dict có key 'bbox'
    Trả về list đã loại bỏ bbox chứa trọn bbox khác
    """
    keep = []
    n = len(bboxes)
    removed_idx = set()
    for i in range(n):
        box_i = bboxes[i]['bbox']
        for j in range(n):
            if i == j: continue
            box_j = bboxes[j]['bbox']
            if is_box_inside(box_i, box_j, margin):
                # Nếu box_i chứa hoàn toàn box_j thì loại box_i (bbox lớn)
                removed_idx.add(i)
                break
    for i in range(n):
        if i not in removed_idx:
            keep.append(bboxes[i])
    return keep


def sort_bboxes_linewise(bboxes, y_thresh=15):
    """
    bboxes: list of dict, mỗi dict có 'bbox' = [x1, y1, x2, y2]
    y_thresh: khoảng cách y để coi là cùng 1 dòng (pixel)
    """
    # Gán index cho mỗi bbox để giữ thứ tự sau sort
    for i, info in enumerate(bboxes):
        info['_idx'] = i

    # Sort bboxes: trước theo y1, rồi x1
    bboxes = sorted(bboxes, key=lambda b: (b['bbox'][1], b['bbox'][0]))

    # Gom thành các dòng
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

    # Mỗi dòng đã sort trái sang phải
    # Flatten lại cho ra thứ tự text hoàn chỉnh
    final_order = []
    for line in lines:
        # Nếu muốn sort lại theo x1 cho chắc (có thể bỏ nếu đã sort ở trên)
        line = sorted(line, key=lambda b: b['bbox'][0])
        final_order.extend(line)
    return final_order

def remove_bboxes_in_rect(infos, rect_x1, rect_y1, rect_x2, rect_y2):
    """
    Loại bỏ bbox nằm hoàn toàn trong vùng hình chữ nhật [rect_x1, rect_y1, rect_x2, rect_y2]
    infos: list dict, mỗi dict có key 'bbox'
    """
    filtered = []
    for info in infos:
        x1, y1, x2, y2 = info['bbox']
        if x1 >= rect_x1 and y1 >= rect_y1 and x2 <= rect_x2 and y2 <= rect_y2:
            continue  # bbox nằm hoàn toàn trong vùng loại bỏ
        filtered.append(info)
    return filtered

def remove_bboxes_in_y_range(infos, lo_bound, up_bound):
    """
    Loại bỏ bbox mà y1 và y2 đều nằm trong [lo_bound, up_bound]
    infos: list dict, mỗi dict có key 'bbox'
    """
    filtered = []
    for info in infos:
        y1 = info['bbox'][1]
        y2 = info['bbox'][3]
        if lo_bound <= y1 <= up_bound and lo_bound <= y2 <= up_bound:
            continue  # bỏ bbox này
        filtered.append(info)
    return filtered

for _video_path in tqdm.tqdm(sorted(args.root_videos.glob("*.mp4"))):
    _video_id = _video_path.stem
    output_file = args.output / _video_id / "ocr_parseq.json"
    # Nếu file đã tồn tại thì skip luôn
    if output_file.exists():
        print(f"Skip {_video_id} (output already exists)")
        continue
    
    cap = cv2.VideoCapture(str(_video_path))
    
    with open(args.root_bboxes / (_video_id + ".json"), 'r') as fi:
        _video_dic = json.load(fi)
    _video_dic = dict(sorted(_video_dic.items(), key=lambda x: int(x[0])))
    
    _video_res = {}
    for _frame_id, _infos in tqdm.tqdm(_video_dic.items()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(_frame_id))
        ret, frame = cap.read()
        
        _infos = remove_bboxes_in_rect(infos=_infos, rect_x1=1040, rect_y1=55, rect_x2=1190, rect_y2=120)
        _infos = remove_bboxes_in_y_range(_infos, lo_bound=655, up_bound=690)
        _infos = remove_containing_boxes(_infos, margin=5)
        _infos = sort_bboxes_linewise(_infos, y_thresh=15)
        
        text = ""
        for _info in _infos:
            x1, y1, x2, y2 = map(int, _info['bbox'])
            score = _info['score']
            
            crop_img = frame[y1:y2, x1:x2]
            crop_img = Image.fromarray(crop_img)
            crop_img = img_transform(crop_img).unsqueeze(0).to(args.device)
            logits = model(crop_img)
            logits.shape  # torch.Size([1, 26, 95]), 94 characters + [EOS] symbol

            # Greedy decoding
            pred = logits.softmax(-1)
            label, confidence = model.tokenizer.decode(pred)
            # print(f"{label[0]} - x1: {x1} - y1: {y1} - x2: {x2} y2 {y2}")
            if (len(label[0]) <= 1): continue
            text = text + " " + label[0]
        _video_res[_frame_id] = text
    
    cap.release()
    with open(output_file, 'w') as f:
        json.dump(_video_res, f)
    
