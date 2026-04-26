# ML_Practical/exp7_pca.py
# Experiment 7: To implementation of Dimensionality Reduction using Principal Component Analysis (PCA) to explain required amount of variance from available features in Python

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load high-dimensional dataset (4 features)
data = load_iris()
X = data.data
y = data.target

# PCA Implementation (Reduce to 2 features)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Output
print("--- Principal Component Analysis (PCA) ---")
print(f"Original Shape: {X.shape}")
print(f"Reduced Shape: {X_pca.shape}")
print(f"Explained Variance Ratio per Component: {pca.explained_variance_ratio_}")
print(f"Total Variance Explained: {np.sum(pca.explained_variance_ratio_):.4f}")

# Visualization
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Dimensionality Reduction on Iris Dataset')
plt.show()