from es_module import search_by_ocr, search_by_asr, get_text_by_frame

video_id = "L25_V084"
frame_id = 31792
res = get_text_by_frame(video_id=video_id, frame_id=frame_id)
print(res)

# res = search_by_asr('feo hcl ag', 10) 
# for video_id, frame_id, score in res:
#     print(f"__{frame_id}__")
#     print(f"__{video_id}__")

# from elasticsearch import Elasticsearch

# es = Elasticsearch("http://elasticsearch:9200")

# response = es.search(
#     index="aic2025",
#     body={
#         "size": 5,
#         "query": {
#             "function_score": {
#                 "query": {"match_all": {}},
#                 "random_score": {"seed": 42}
#             }
#         },
#         "_source": ["video_id", "frame_id", "ocr_text", "asr_text"]
#     }
# )

# for doc in response["hits"]["hits"]:
#     src = doc["_source"]
#     print(f"{src['video_id']} | {src['frame_id']}")
#     print("OCR:", src.get("ocr_text", "")[:100])
#     print("ASR:", src.get("asr_text", "")[:100])
#     print("-" * 40)