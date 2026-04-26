# ML_Practical/exp2_logistic_regression.py
# Experiment 2: To implementation of Logistic Regression

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generating a simple 2-class dataset
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)

# Model Implementation
model = LogisticRegression()
model.fit(X, y)

# Output
print("--- Logistic Regression ---")
print(f"Accuracy: {model.score(X, y):.4f}")

# Visualization (Decision Boundary)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='bwr')
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='bwr')
plt.title("Logistic Regression Decision Boundary")
plt.show()