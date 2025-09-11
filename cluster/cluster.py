import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import shutil
import time
from tqdm import tqdm

# Set paths
root_folder = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/merge")
output_folder = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/group/600gr")
output_folder.mkdir(parents=True, exist_ok=True)

# Load vectors
all_vectors = []
image_info = []  # Keep track of image paths and video IDs

start_time = time.time()

videos = sorted(root_folder.iterdir())

for _video in tqdm(videos):
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

# Check if vectors are loaded
if not all_vectors:
    raise ValueError("No vectors found. Check if your .npz files contain a 'feature' key.")

X = np.vstack(all_vectors)
print(f"Loaded {X.shape[0]} vectors with dimension {X.shape[1]}")

# Normalize for cosine similarity
X = normalize(X)

# Clustering with MiniBatchKMeans (no PCA)
print("Clustering with MiniBatchKMeans...")
n_clusters = 600  # tune as needed
clusterer = MiniBatchKMeans(n_clusters=n_clusters, batch_size=10000, random_state=42, verbose=1)
labels = clusterer.fit_predict(X)

# Save labels
np.save("cluster_labels_600.npy", labels)
print(f"Assigned {n_clusters} clusters using MiniBatchKMeans.")

# Copy images into group folders
for label, (img_path, video_id, original_name) in tqdm(
    zip(labels, image_info), total=len(labels), desc="Copying images"
):
    cluster_dir = output_folder / f"group_{label}"
    cluster_dir.mkdir(parents=True, exist_ok=True)

    new_name = f"{video_id}-{original_name}"
    destination = cluster_dir / new_name

    if img_path.exists():
        shutil.copy(img_path, destination)
    else:
        print(f"Missing image: {img_path}")

# t-SNE visualization (sample only, since t-SNE is heavy)
sample_size = min(3000, X.shape[0])
X_vis = X[:sample_size]
labels_vis = labels[:sample_size]

print("Running t-SNE on a sample for visualization...")
Y = TSNE(n_components=2, perplexity=50, random_state=42).fit_transform(X_vis)

plt.figure(figsize=(10, 8))
plt.scatter(Y[:, 0], Y[:, 1], c=labels_vis, cmap='tab20', s=2)
plt.title("MiniBatchKMeans Clustering Visualization (no PCA)")
plt.savefig("cluster_labels_600.png", dpi=300)
print("Saved clustering plot to kmeans_clusters.png")

print("total time: ", time.time() - start_time)
