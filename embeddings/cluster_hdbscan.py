import numpy as np
from sklearn.preprocessing import normalize
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import shutil
import time
from tqdm import tqdm
import hdbscan

# -------------------------
# Set paths
# -------------------------
root_folder = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/batch1")
output_folder = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/group/batch1_hdbscan")
output_folder.mkdir(parents=True, exist_ok=True)

# -------------------------
# Load vectors
# -------------------------
all_vectors = []
image_info = []

start_time = time.time()

videos = sorted(root_folder.iterdir())

for _video in tqdm(videos, desc="Loading videos"):
    _keyframes = _video / "keyframes"
    _vectors = _video / "vector_file"
    video_id = _video.name

    for _npy_file in sorted(_vectors.glob("*.npz")):
        keyframe_name = _npy_file.stem + '.webp'
        keyframe_path = _keyframes / keyframe_name

        with np.load(_npy_file) as data:
            if "feature" in data:
                all_vectors.append(data["feature"])
                image_info.append((keyframe_path, video_id, keyframe_name))
            else:
                print(f"Warning: No 'feature' key in {_npy_file}")

if not all_vectors:
    raise ValueError("No vectors found. Check your .npz files.")

X = np.vstack(all_vectors)
print(f"Loaded {X.shape[0]} vectors with dimension {X.shape[1]}")

# -------------------------
# Normalize features
# -------------------------
X = normalize(X)

# -------------------------
# Clustering with HDBSCAN
# -------------------------
print("Clustering with HDBSCAN...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', core_dist_n_jobs=-1)
labels = clusterer.fit_predict(X)

# Save labels
np.save("cluster_labels_hdbscan.npy", labels)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"HDBSCAN assigned {n_clusters} clusters (label -1 = noise)")

# -------------------------
# Copy clustered images
# -------------------------
for label, (img_path, video_id, original_name) in tqdm(
    zip(labels, image_info), total=len(labels), desc="Copying images"
):
    if label == -1:
        cluster_dir = output_folder / f"group_noice"
    else:
        cluster_dir.mkdir(parents=True, exist_ok=True)

    new_name = f"{video_id}-{original_name}"
    destination = cluster_dir / new_name

    if img_path.exists():
        shutil.copy(img_path, destination)
    else:
        print(f"Missing image: {img_path}")

# -------------------------
# t-SNE visualization (sample)
# -------------------------
sample_size = min(3000, X.shape[0])
X_vis = X[:sample_size]
labels_vis = labels[:sample_size]

print("Running t-SNE on a sample for visualization...")
Y = TSNE(n_components=2, perplexity=50, random_state=42).fit_transform(X_vis)

plt.figure(figsize=(10, 8))
plt.scatter(Y[:, 0], Y[:, 1], c=labels_vis, cmap='tab20', s=2)
plt.title("HDBSCAN Clustering Visualization (t-SNE)")
plt.savefig("hdbscan_clusters.png", dpi=300)
print("Saved clustering plot to hdbscan_clusters.png")

print("Total time:", time.time() - start_time)
