import torch
from pymilvus import Collection, connections
import torch.nn.functional as F
import open_clip
from pydantic import BaseModel
import time
from collections import defaultdict


# === CONFIG === 
COLLECTION_NAME = 'AIC25_fullbatch1'
DIMENSION = 1024
TOP_K = 1000
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
    ])
start_time = time.time()

# === SEARCH FOR EACH QUERY ===
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

# === ALIGN AND SCORE CHAINS ===
best_chain = None
best_score = -1e9
TOP_K_CHAINS = 3
all_chains = []

for vid, stage_hits in video_groups.items():
    if any(len(s) == 0 for s in stage_hits):
        continue

    tensor_stages = []
    for stage in stage_hits:
        stage_sorted = sorted(stage)
        fids = torch.tensor([f[0] for f in stage_sorted], device=device)
        scores = torch.tensor([f[1] for f in stage_sorted], device=device)
        tensor_stages.append((fids, scores, stage_sorted))

    n_stages = len(tensor_stages)
    dp_scores = [None] * n_stages
    dp_paths = [None] * n_stages

    dp_scores[0] = tensor_stages[0][1]
    dp_paths[0] = [[i] for i in range(len(tensor_stages[0][1]))]

    for i in range(1, n_stages):
        prev_fids, prev_scores, _ = tensor_stages[i - 1]
        curr_fids, curr_scores, _ = tensor_stages[i]

        diff = curr_fids[:, None] - prev_fids[None, :]
        valid = (diff > 0) & (diff <= MAX_FRAME_GAP // len(query.Queries))

        # decay = (MAX_FRAME_GAP - diff) / MAX_FRAME_GAP
        decay = torch.sigmoid((MAX_FRAME_GAP / 2 - diff.float()) / 50)
        temp_score = dp_scores[i - 1][None, :] + curr_scores[:, None] * decay
        temp_score = torch.where(valid, temp_score, torch.full_like(temp_score, -1e9))

        max_vals, max_idxs = temp_score.max(dim=1)
        dp_scores[i] = max_vals
        dp_paths[i] = [dp_paths[i - 1][j.item()] + [k] for j, k in zip(max_idxs, range(len(curr_fids)))]

    for idx, score in enumerate(dp_scores[-1]):
        for stage_i, path in enumerate(dp_paths[-1][idx]):
            all_chains.append((score.item(), tensor_stages[stage_i][2][path], vid))

# Sort chains across all videos
all_chains.sort(key=lambda x: -x[0])

# === SHOW RESULTS ===
end_time = time.time()
print(f"\nTop {TOP_K_CHAINS} Matching Frame Chains:")
for rank, (score, (frame_id, distance_score, path), vid) in enumerate(all_chains[:10], 1):
    weighted_score = 0.1 * (len(all_chains) - rank) / len(all_chains) +  0.7 * score * (len(all_chains) - rank) / len(all_chains) + 0.2 * distance_score
    print(f"  Stage {i+1}: {path} | Frame ID: {frame_id} | Distance: {weighted_score:.4f}")

print(f"\nTotal time: {end_time - start_time:.2f}s")