import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import time
import warnings

warnings.filterwarnings("ignore")
np.random.seed(0)
X, y = make_blobs(
    n_samples=5000, centers=[[4, 4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9
)
# optional: standardize the features
plt.scatter(X[:, 0], X[:, 1], s=50, cmap="viridis")
plt.title("Generated Blobs for Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
#
k_means = KMeans(n_clusters=4, init="k-means++", n_init=12)
k_means.fit(X)
k_means_labels = k_means.labels_
print(f"{"="*40}k_means_labels{"="*40}")
print(k_means_labels)
k_means_cluster_centers = k_means.cluster_centers_

# visualize the clusters
print(f"{"="*40}k_means_cluster_centers{"="*40}")
print(k_means_cluster_centers)
print("=" * 100)
fig = plt.figure(figsize=(6, 4))
print(set(k_means_labels))
colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(
        X[my_members, 0], X[my_members, 1], "w", markerfacecolor=col, marker=".", ms=10
    )
    ax.plot(
        cluster_center[0],
        cluster_center[1],
        "o",
        markerfacecolor=col,
        markeredgecolor="k",
        markersize=6,
    )
ax.set_title("KMeans")

plt.show()
