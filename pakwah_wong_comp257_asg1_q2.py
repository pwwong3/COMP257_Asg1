# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:11:50 2024

@author: Pak Wah Wong
"""
#import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

# 1. Generate Swiss roll dataset.
X, y = make_swiss_roll(n_samples=1000, noise=0.04, random_state=0)

# 2. Plot the resulting generated Swiss roll dataset.
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
fig.add_axes(ax)
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
ax.set_title("Swiss Roll")
plt.show()

# 3. Use Kernel PCA (kPCA) with linear kernel, a RBF kernel, and a sigmoid kernel
linear_pca = KernelPCA(n_components=2, kernel="linear")
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
sigmoid_pca = KernelPCA(n_components=2, kernel="sigmoid", gamma=0.001)

X_reduced_linear = linear_pca.fit_transform(X)
X_reduced_rbf = rbf_pca.fit_transform(X)
X_reduced_sigmoid = sigmoid_pca.fit_transform(X)

# 4. Plot the kPCA results of applying the linear kernel, a RBF kernel, and a sigmoid kernel
fig, (linear_pca_ax, rbf_pca_ax, sigmoid_pca_ax) = plt.subplots(
    ncols=3, figsize=(14, 4)
)

linear_pca_ax.scatter(X_reduced_linear[:, 0], X_reduced_linear[:, 1], c=y)
linear_pca_ax.set_title("Projection of data\n using linear kernel")

rbf_pca_ax.scatter(X_reduced_rbf[:, 0], X_reduced_rbf[:, 1], c=y)
rbf_pca_ax.set_title("Projection of data\n using rbf kernel")

sigmoid_pca_ax.scatter(X_reduced_sigmoid[:, 0], X_reduced_sigmoid[:, 1], c=y)
sigmoid_pca_ax.set_title("Projection of data\n using sigmoid kernel")

plt.show()

# Using kPCA and a kernel of your choice, apply Logistic Regression for classification.
# Use GridSearchCV to find the best kernel and gamma value for kPCA in order to get the best
# classification accuracy at the end of the pipeline. Print out best parameters found by GridSearchCV.

clf = Pipeline([
  ("kpca", KernelPCA(n_components=2)),
  ("log_reg", LogisticRegression())
])

param_grid = [{
  "kpca__gamma": np.linspace(0.03, 0.05, 10),
  "kpca__kernel": ["rbf", "sigmoid", "linear"],
}]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y < 10)
print(grid_search.best_params_)

# Plot the results from using GridSearchCV.
X_kpca = grid_search.best_estimator_["kpca"].transform(X)
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y<10, cmap=plt.cm.Spectral)
plt.show()