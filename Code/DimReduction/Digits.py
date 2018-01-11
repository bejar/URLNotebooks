"""
.. module:: Digits

Digits
*************

:Description: Dimensionality Reduction applied to the digits dataset


:Authors: bejar
    

:Version: 

:Created on: 07/09/2015 13:53 

"""

__author__ = 'bejar'


from sklearn.datasets import load_digits
from pylab import *
#import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

digits = load_digits()

# PCA transformation

pca = PCA()
fdata = pca.fit_transform(digits['data'])

print(pca.explained_variance_ratio_)

fig = plt.figure()

plt.plot(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.scatter(fdata[:, 0], fdata[:, 1], zs= fdata[:, 2], c=digits['target'],s=100)

plt.show()


# ISOMAP

from sklearn.manifold import Isomap
iso = Isomap(n_components=3, n_neighbors=15)
fdata = iso.fit_transform(digits['data'])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.scatter(fdata[:, 0], fdata[:, 1], zs= fdata[:, 2], c=digits['target'], s=100)

plt.show()


# LLE

from sklearn.manifold import LocallyLinearEmbedding
lle = LocallyLinearEmbedding(n_neighbors=15, n_components=3, method='modified')
fig = plt.figure()
fdata = lle.fit_transform(digits['data'])
ax = fig.add_subplot(111, projection='3d')

plt.scatter(fdata[:, 0], fdata[:, 1], zs= fdata[:, 2], c=digits['target'], s=100)

plt.show()

# MDS

from sklearn.manifold import MDS

mds = MDS(n_components=3)
fig = plt.figure()
fdata = mds.fit_transform(digits['data'])
ax = fig.add_subplot(111, projection='3d')

plt.scatter(fdata[:, 0], fdata[:, 1], zs= fdata[:, 2], c=digits['target'], s=100)

plt.show()
