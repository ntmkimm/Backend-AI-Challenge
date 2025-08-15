import torch
from pymilvus import Collection, connections
import torch.nn.functional as F
import open_clip

# === CONFIG ===
COLLECTION_NAME = 'AIC25_fullbatch1'
DIMENSION = 1024
TOP_K = 10
host = "192.168.20.156"
port = "19530"

# === LOAD COLLECTION ===
connections.connect(host=host, port=port)
collection = Collection(name=COLLECTION_NAME)
collection.load()

# === LOAD CLIP MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _, _ = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
tokenizer = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')
model = model.to(device)
model.eval()

# === ENCODE TEXT ===
while True:
    print("Type one question:")
    TEXT_QUERY = input()
    text_tokens = tokenizer([TEXT_QUERY]).to(device)

    with torch.no_grad():
        text_embedding = model.encode_text(text_tokens)
        text_embedding = F.normalize(text_embedding, p=2, dim=-1).cpu().tolist()

    # === SEARCH ===
    results = collection.search(
        data=text_embedding,
        anns_field="clip_embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=TOP_K,
        output_fields=["filepath"],
    )

    # === SHOW RESULTS ===
    for i, hit in enumerate(results[0]):
        path = hit.entity.get("filepath").split('/')
        filename = path[-1].split('.')[0] if path else 'unknown'
        video_id = path[-3] if path else 'unknown'
        print(filename)
        print(video_id)
        print(f"Top {i+1} Match: {hit.entity.get('filepath')} (score: {hit.distance:.4f})")
