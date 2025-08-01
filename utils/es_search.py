from es_module import search_by_ocr, search_by_asr


res = search_by_ocr('hát bội', 10) 
for video_id, frame_id, score, path in res:
    print(frame_id)