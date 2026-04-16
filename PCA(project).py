import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
sccaler = StandardScaler()
X_scaled = sccaler.fit_transform(X)
print(iris.target_names)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(X_scaled)
plt.figure()
colors = ["navy", "turquoise", "darkorange"]
lw = 2
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        x_pca[iris.target == i, 0],
        x_pca[iris.target == i, 1],
        color=color,
        alpha=0.8,
        lw=lw,
        label=target_name,
    )
plt.title(
    "PCA 2-dimensional reduction of IRIS dataset",
)
plt.xlabel(
    "PC1",
)
plt.ylabel(
    "PC2",
)
plt.legend(
    loc="best",
    shadow=False,
    scatterpoints=1,
)
plt.grid(True)
plt.show()
separator = "=" * 40
print(f"{separator} components {separator}")
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_

plt.figure(figsize=(10, 6))
plt.bar(
    x=range(1, len(explained_variance_ratio) + 1),
    height=explained_variance_ratio,
    alpha=1,
    align="center",
    label="PC explained variance ratio",
)
plt.ylabel("Explained Variance Ratio")
plt.xlabel("Principal Components")
plt.title("Explained Variance by Principal Components")

cumulative_variance = np.cumsum(explained_variance_ratio)
plt.step(
    range(1, 5),
    cumulative_variance,
    where="mid",
    linestyle="--",
    lw=3,
    color="red",
    label="Cumulative Explained Variance",
)
plt.xticks(range(1, 5))
plt.legend()
plt.grid(True)
plt.show()
