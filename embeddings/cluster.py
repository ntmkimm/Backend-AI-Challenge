import numpy as np
from sklearn.preprocessing import normalize
import hdbscan
from pathlib import Path
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load and stack all embedding vectors
root_folder = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/AIChallenge2025/dataset/full/batch1")
all_vectors = []

for _video in sorted(root_folder.iterdir()):
    _vectors = _video / "vector_file"
    for _npy_file in sorted(_vectors.glob("*.npz")):
        with np.load(_npy_file) as data:
            if "feature" in data:
                x = data["feature"]
                all_vectors.append(x)
            else:
                print(f"Warning: No 'feature' key in {_npy_file}")

# Guard against empty list
if not all_vectors:
    raise ValueError("No vectors found. Check if your .npz files contain a 'feature' key.")

X = np.vstack(all_vectors)
print(f"Loaded {X.shape[0]} vectors with dimension {X.shape[1]}")

# Normalize for cosine similarity
X = normalize(X)

# HDBSCAN clustering
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, metric='euclidean')
labels = clusterer.fit_predict(X)

# Cluster stats
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"Found {n_clusters} clusters.")

# Optional: Save labels
np.save("cluster_labels.npy", labels)

# t-SNE visualization (on a 3k sample)
sample_size = min(3000, X.shape[0])
X_vis = X[:sample_size]
labels_vis = labels[:sample_size]

Y = TSNE(n_components=2, perplexity=50, random_state=42).fit_transform(X_vis)
plt.figure(figsize=(10, 8))
plt.scatter(Y[:, 0], Y[:, 1], c=labels_vis, cmap='tab20', s=2)
plt.title("HDBSCAN Clustering Visualization")
plt.savefig("hdbscan_clusters.png", dpi=300)
print("Saved clustering plot to hdbscan_clusters.png")
