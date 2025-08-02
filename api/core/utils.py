from fastapi import APIRouter, HTTPException, Depends
from typing import List
from models.schemas import Query
from typing import List
from models.schemas import Query
from PIL import Image
from io import BytesIO
import base64
import cv2
import numpy as np
from typing import List
from models.schemas import Query

def get_valid_queries(queries: List[Query]) -> List[Query]:
    res = []
    for q in queries:
        if not q.text and not q.ocr and not q.asr and not q.origin and not q.obj and not q.image:
            continue
        res.append(q)
    return res

def base64_to_cv2_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img

def base64_to_pil_image(base64_str):
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    img = img.convert("RGB")  # Bắt buộc chuyển về RGB cho model encode (an toàn)
    return img