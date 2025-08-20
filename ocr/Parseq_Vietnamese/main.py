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
    root_videos = Path('/mlcv2/Datasets/HCMAI25/batch1/video')
    root_bboxes = Path('/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/backend/ocr/json/batch1_2025')
    output = Path('/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/batch1')
    device = 'cuda'
    batch_size = 36
    num_workers = 1
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

# --- BBox helper functions ---
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

def remove_bboxes_in_rect(infos, rect_x1, rect_y1, rect_x2, rect_y2):
    return [info for info in infos if not (rect_x1 <= info['bbox'][0] and rect_y1 <= info['bbox'][1] and
                                           rect_x2 >= info['bbox'][2] and rect_y2 >= info['bbox'][3])]

def remove_bboxes_in_y_range(infos, lo_bound, up_bound):
    return [info for info in infos if not (lo_bound <= info['bbox'][1] <= up_bound and
                                           lo_bound <= info['bbox'][3] <= up_bound)]

# --- Video Range Config ---
start_video = 'L29_V001'
end_video = 'L31_V001'
start_video2 = 'L26_V001'
end_video2 = 'L26_V337'
start_video1 = 'L28_V009'
end_video1 = 'L29_V001'

print("start_video: ", start_video)
print("end_video: ", end_video)

video_files = []
for _video_path in sorted(args.root_videos.glob("*.mp4")):
    video_name = _video_path.stem
    if not ((start_video2 <= video_name < end_video2) or (start_video1 <= video_name < end_video1)):
        continue
    video_files.append(_video_path)
    
# print("reverse")
# video_files = video_files[::-1]

# --- Process Videos ---
for _video_path in tqdm.tqdm(video_files):
    _video_id = _video_path.stem
    print("Processing video:", _video_id)
    output_file = args.output / _video_id / "ocr_parseq.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(_video_path))
    bbox_file = args.root_bboxes / (_video_id + ".json")
    if not bbox_file.exists():
        continue

    with open(bbox_file, 'r') as fi:
        _video_dic = json.load(fi)

    _video_dic = dict(sorted(_video_dic.items(), key=lambda x: int(x[0])))
    _video_res = {}

    for _frame_id, _infos in tqdm.tqdm(_video_dic.items()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(_frame_id))
        ret, frame = cap.read()
        if not ret:
            continue

        _infos = remove_bboxes_in_y_range(_infos, lo_bound=655, up_bound=690)
        _infos = remove_containing_boxes(_infos, margin=5)
        _infos = sort_bboxes_linewise(_infos, y_thresh=15)

        text = ""
        h, w = frame.shape[:2]
        crop_imgs = []
        _id_batch = []

        for _id_info, _info in enumerate(_infos):
            x1, y1, x2, y2 = map(int, _info['bbox'])
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            if x1 >= x2 or y1 >= y2:
                continue

            crop_img = frame[y1:y2, x1:x2]
            if crop_img.size == 0:
                continue
            crop_imgs.append(Image.fromarray(crop_img))
            _id_batch.append(_id_info)

            if len(crop_imgs) == args.batch_size:
                with torch.no_grad():
                    batch = torch.stack([img_transform(img) for img in crop_imgs]).to(args.device)
                    logits = model(batch)
                    pred = logits.softmax(-1).detach().cpu()
                    labels, confidences = model.tokenizer.decode(pred)

                for _idd, _label, _conf in zip(_id_batch, labels, confidences):
                    _infos[_idd]["text"] = _label
                    # _infos[_idd]["score_parseq"] = float(_conf)
                    # print(_conf)

                text += " " + " ".join([l.lower() for l in labels])
                crop_imgs, _id_batch = [], []
                del batch, logits, pred
                torch.cuda.empty_cache()

        if crop_imgs:
            with torch.no_grad():
                batch = torch.stack([img_transform(img) for img in crop_imgs]).to(args.device)
                logits = model(batch)
                pred = logits.softmax(-1).detach().cpu()
                labels, confidences = model.tokenizer.decode(pred)

            for _idd, _label, _conf in zip(_id_batch, labels, confidences):
                _infos[_idd]["text"] = _label
                # _infos[_idd]["score_parseq"] = float(_conf)

            text += " " + " ".join([l.lower() for l in labels])
            del batch, logits, pred
            torch.cuda.empty_cache()

        _video_dic[_frame_id] = _infos
        _video_res[_frame_id] = text.lower()

    cap.release()

    out_dir = args.root_bboxes.parent / (args.root_bboxes.name + "_parseq")
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(out_dir / (_video_id + ".json"), "w") as fi:
        json.dump(_video_dic, fi, indent=4)

    with open(output_file, 'w') as f:
        json.dump(_video_res, f, indent=4)
