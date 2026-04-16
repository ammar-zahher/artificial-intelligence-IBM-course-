import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
mean = [0, 0]
cov = [[3, 2], [2, 2]]
X = np.random.multivariate_normal(mean=mean, cov=cov, size=1000)
X1 = X[:, 0]
X2 = X[:, 1]
plt.figure()
plt.scatter(
    X1,
    X2,
    edgecolors="k",
    alpha=0.7,
)
plt.colorbar()
plt.title("visualiz the data")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.grid(True)
plt.show()

pca = PCA(n_components=2)
x_pca = pca.fit_transform(X)
components = pca.components_
separator = "=" * 40
print(f"{separator} components {separator}")
print(components)
pca.explained_variance_ratio_
print(f"{separator} explained_variance_ratio_ {separator}")
print(pca.explained_variance_ratio_)
m = np.dot(X, components[0])
m2 = np.dot(X, components[1])
x_pc1 = m * components[0][0]
y_pc1 = m * components[0][1]
x_pc2 = m2 * components[1][0]
y_pc2 = m2 * components[1][1]
plt.figure()
plt.scatter(X[:, 0], X[:, 1], label="Original Data", ec="k", s=50, alpha=0.6)

# Plot the projections along PC1 and PC2
plt.scatter(
    x_pc1,
    y_pc1,
    c="r",
    ec="k",
    marker="X",
    s=70,
    alpha=0.5,
    label="Projection onto PC 1",
)
plt.scatter(
    x_pc2,
    y_pc2,
    c="b",
    ec="k",
    marker="X",
    s=70,
    alpha=0.5,
    label="Projection onto PC 2",
)
plt.title(
    "Linearly Correlated Data Projected onto Principal Components",
)
plt.xlabel(
    "Feature 1",
)
plt.ylabel(
    "Feature 2",
)
plt.legend()
plt.grid(True)
plt.axis("equal")
plt.show()
