# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 16:11:50 2024

@author: Pak Wah Wong
"""
#import the necessary libraries
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

# 1. Retrieve and load the mnist_784 dataset of 70,000 instances.
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X = mnist.data
y = mnist.target

# 2. Display each digit
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(25):
    ax = fig.add_subplot(5, 5, i+1, xticks=[], yticks=[])
    ax.imshow(X[i].reshape(28,28), cmap=plt.cm.bone, interpolation='nearest')
plt.show()

# 3. Use PCA to retrieve the 1st and 2nd principal component and output their explained variance ratio.
n_components = 2  # Number of principal components
#Create the instance of PCA
pca = PCA(n_components=n_components)
#Fit the training data
pca.fit(X)
# explained variance
print("Explained variance ratio: ", pca.explained_variance_ratio_)

# 4. Plot the projections of the 1st and 2nd principal component onto a 1D hyperplane.
X_pca = pca.transform(X)

# Plot the principal components of MNIST Dataset
fig, (pc1_ax, pc2_ax) = plt.subplots(
    nrows=2, figsize=(14, 8)
)
for i in range(10):
    pc1_ax.plot(X_pca[y == str(i), 0], np.zeros_like(X_pca[y == str(i), 0]), label=str(i))
    pc1_ax.legend(loc="upper right")
    pc1_ax.set_xlabel('Principal Component 1')
    pc2_ax.plot(X_pca[y == str(i), 1], np.zeros_like(X_pca[y == str(i), 1]), label=str(i))
    pc2_ax.legend(loc="upper right")
    pc2_ax.set_xlabel('Principal Component 2')
plt.show()

# 5. Use Incremental PCA to reduce the dimensionality of the MNIST dataset down to 154 dimensions.
n_components5 = 154  # Number of principal components
pca5 = PCA(n_components=n_components5)
X_pca5 = pca5.fit_transform(X)
X_recovered = pca5.inverse_transform(X_pca5)

fig5 = plt.figure(figsize=(10,2))
fig5.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(10):
    ax = fig5.add_subplot(2, 10, i+1, xticks=[], yticks=[])
    ax.imshow(X[i].reshape(28,28), cmap=plt.cm.bone, interpolation='nearest')
    ax = fig5.add_subplot(2, 10, i+11, xticks=[], yticks=[])
    ax.imshow(X_recovered[i].reshape(28,28), cmap=plt.cm.bone, interpolation='nearest')
plt.show()


