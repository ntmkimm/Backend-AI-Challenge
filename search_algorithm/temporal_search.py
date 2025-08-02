import torch
from pymilvus import Collection, connections
import torch.nn.functional as F
import open_clip
from pydantic import BaseModel
import time
from collections import defaultdict

# === CONFIG ===
COLLECTION_NAME = 'AIC25_fullbatch1'
# COLLECTION_NAME = 'testti'
DIMENSION = 1024
TOP_K = 500
host = "192.168.20.156"
port = "19530"
MAX_FRAME_GAP = 750

class TextQuery(BaseModel):
    Queries: list[str]

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

# === INPUT QUERIES ===
query = TextQuery(Queries=[
    "there are many people sitting on the beds",
    "a hand holds another hand which have purple sign",
    # "a women is sitting on the bed"
])

start_time = time.time()

# === Step 1: Batch Tokenize and Encode Queries ===
with torch.no_grad():
    tokens = tokenizer(query.Queries).to(device)
    embeddings = model.encode_text(tokens)
    embeddings = F.normalize(embeddings, p=2, dim=-1)  # shape: [num_queries, 1024]

end_time = time.time()
print(end_time - start_time)

all_answers = [[] for _ in range(len(embeddings))]

# === Iterate over each embedding ===
for query_idx, embedding in enumerate(embeddings.cpu().tolist()):
    iterator = collection.search_iterator(
        data=[embedding], 
        anns_field="clip_embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=TOP_K,
        batch_size=200,
        output_fields=["filepath", "frame_id", "video_id"],
    )

    while True:
        hits = iterator.next()
        if not hits:
            iterator.close()
            break
        all_answers[query_idx].extend(hits)  # Add to correct query

end_time = time.time()
print(end_time - start_time)

# === GROUP BY VIDEO ID ===
video_groups = defaultdict(lambda: [[] for _ in range(len(query.Queries))])
for stage_idx, hits in enumerate(all_answers):
    for h in hits:
        video_groups[h.entity["video_id"]][stage_idx].append((int(h.entity["frame_id"]), h.distance, h.entity["filepath"]))

# === TEMPORAL SCORING (NO CHAIN) ===
final_results = []

for vid, stage_hits in video_groups.items():
    if any(len(stage) == 0 for stage in stage_hits):
        continue

    # Convert each stage to tensors
    tensor_stages = []
    for stage in stage_hits:
        stage_sorted = sorted(stage)
        fids = torch.tensor([x[0] for x in stage_sorted], device=device)
        scores = torch.tensor([x[1] for x in stage_sorted], device=device)
        tensor_stages.append((fids, scores, stage_sorted))

    base_fids, base_scores, base_raw = tensor_stages[0]
    final_scores = base_scores.clone()

    for i in range(1, len(tensor_stages)):

        curr_fids, curr_scores, _ = tensor_stages[i]
        diff = curr_fids[:, None] - base_fids[None, :]
        valid = (diff > 0) & (diff <= MAX_FRAME_GAP)

        # decay = (MAX_FRAME_GAP - diff.float()) / MAX_FRAME_GAP
        decay = torch.sigmoid((MAX_FRAME_GAP / 2 - diff.float()) / 30)
        boost = curr_scores[:, None] * decay
        boost = torch.where(valid, boost, torch.zeros_like(boost))

        num_valid = valid.sum(dim=0).clamp(min=1)
        final_scores += boost.sum(dim=0) / num_valid

    for i in range(len(base_fids)):
        frame_id, dist, path = base_raw[i]
        final_results.append((final_scores[i].item(), path, frame_id, vid))

# === SORT AND DISPLAY RESULTS ===
final_results.sort(key=lambda x: -x[0])
end_time = time.time()

print(f"\nTop 10 Matches (Non-Chain):")
for i, (score, path, frame_id, vid) in enumerate(final_results[:10]):
    print(f"Top {i+1}: {path} | Frame ID: {frame_id} | Score: {score:.4f} | Video ID: {vid}")

print(f"\nTotal time: {end_time - start_time:.2f}s")
