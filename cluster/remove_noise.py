from pathlib import Path
from tqdm import tqdm

ROOT = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge")
ROOT_GROUP = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/group/merge_new_200")

# default_cluster = [2, 17, 39, 56, 57, 60, 63, 65, 78, 79, 90, 92, 93, 95, 98, 111, 113, 121, 124, 132, 146, 147, 150, 161, 166, 169, 176, 192]
# default_cluster += [131]

default_cluster = [15, 18, 32, 70, 75, 82, 83, 90, 96, 123, 138, 151, 152, 173, 174, 175, 176, 181, 185, 197, ]

_imgs = []
for _gr in tqdm(default_cluster, desc="prepare cluster to delete"):
    _gr_path = ROOT_GROUP / ("group_" + str(_gr))
    for _img in sorted(_gr_path.glob("*.webp")):
        _img_data = {}
        tmp = _img.stem.split("-")
        _img_data["video_id"] = tmp[0]
        _img_data["frame_id"] = int(tmp[1].split("_")[1])
        _imgs.append(_img_data)

print("milvus")
from pymilvus import connections, Collection

connections.connect("default", host="192.168.20.156", port="19530")

collection_openclip = Collection("AIC25_openclip")
collection_beit3 = Collection("AIC25_beit3")
# collection_siglip2 = Collection("AIC25_siglip2")

before_openclip = collection_openclip.num_entities
before_beit3 = collection_beit3.num_entities
# before_siglip2 = collection_siglip2.num_entities

print(before_openclip)
print(before_beit3)
# print(before_siglip2)

t = input("confirm? [y/n]")
if t != 'y': exit()

for _img in tqdm(_imgs):
    expr = f"video_id == '{_img['video_id']}' and frame_id == {_img['frame_id']}"
    collection_openclip.delete(expr=expr)
    collection_beit3.delete(expr=expr)
    # collection_siglip2.delete(expr=expr)
    
collection_openclip.compact()
collection_beit3.compact()
# collection_siglip2.compact()

after_openclip = collection_openclip.num_entities
after_beit3 = collection_beit3.num_entities
# after_siglip2 = collection_siglip2.num_entities

print(f"AIC25_openclip: {before_openclip - after_openclip} records removed")
print(f"AIC25_beit3:     {before_beit3 - after_beit3} records removed")
# print(f"AIC25_siglip2:     {before_siglip2 - after_siglip2} records removed")
    
# print("elastic")
# from elasticsearch import Elasticsearch

# es = Elasticsearch("http://elasticsearch:9200")
# shoulds = [
#     {"bool": {"must": [
#         {"match": {"video": _img["video"]}},
#         {"match": {"frame_id": _img["frame_id"]}}
#     ]}}
#     for _img in _imgs
# ]

# resp = es.delete_by_query(
#     index="aic2025",
#     body={"query": {"bool": {"should": shoulds}}},
#     refresh=True
# )
# print(f"Documents deleted: {resp['deleted']}")

# print("polar")
# OBJECT_DATABASE = "/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge/objects.parquet"
# OUTPUT_DATABASE = str(Path(OBJECT_DATABASE).with_name("objects.parquet"))
# db = pl.read_parquet(OBJECT_DATABASE)
# before_count = db.height

# filtered = db.join(_imgs, on=["video_id", "frame_id"], how="anti")
# filtered.write_parquet(OUTPUT_DATABASE)
# after_count = filtered.height
# print(f"Deleted records:  {before_count - after_count}")

