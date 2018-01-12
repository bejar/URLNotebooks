"""
.. module:: Digits

Digits
*************

:Description: Dimensionality Reduction applied to the digits dataset


:Authors: bejar
    

:Version: 

:Created on: 07/09/2015 13:53 

"""

from __future__ import print_function
from sklearn.datasets import load_digits
from pylab import *
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

__author__ = 'bejar'

def plot_figure(fdata, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.suptitle(title, fontsize=32)
    plt.scatter(fdata[:, 0], fdata[:, 1], zs=fdata[:, 2], c=digits['target'], s=100, cmap='Paired')

    plt.show()

if __name__ == '__main__':
    digits = load_digits()

    # PCA transformation
    pca = PCA()
    fdata = pca.fit_transform(digits['data'])

    fig = plt.figure()
    plt.plot(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    plt.show()

    plot_figure(fdata, 'PCA')

    # ISOMAP
    from sklearn.manifold import Isomap

    iso = Isomap(n_components=3, n_neighbors=15)
    fdata = iso.fit_transform(digits['data'])
    plot_figure(fdata, 'ISOMAP')

    # LLE
    from sklearn.manifold import LocallyLinearEmbedding

    lle = LocallyLinearEmbedding(n_neighbors=15, n_components=3, method='modified')
    fdata = lle.fit_transform(digits['data'])
    plot_figure(fdata, 'LLE')

    # MDS
    from sklearn.manifold import MDS

    mds = MDS(n_components=3)
    fdata = mds.fit_transform(digits['data'])
    plot_figure(fdata, 'MDS')

    # TSNE
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=3, perplexity=25, early_exaggeration=100)
    fdata = tsne.fit_transform(digits['data'])
    plot_figure(fdata, 't-SNE')
