from es_module import search_by_ocr, search_by_asr, get_text_by_frame

video_id = "L01_V002"
frame_id = "0"
res = get_text_by_frame(video_id=video_id, frame_id=frame_id)
print(res)
# res = search_by_asr('hát bội', 10) 
# for video_id, frame_id, score, path in res:
#     print(frame_id)
#     print(video_id)
#     print(path)