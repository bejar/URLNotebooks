"""
.. module:: TseriesClust

TseriesClust
*************

:Description: TseriesClust

    

:Authors: bejar
    

:Version: 

:Created on: 22/12/2017 10:27 

"""

from scipy.spatial import distance
from scipy.signal import resample
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
import numpy as np
from kemlglearn.cluster import KMedoidsFlexible
from fastdtw import fastdtw




def sel(n, i, j):
    return n * i - (i * (i+1)//2) + (j - i) - 1

__author__ = 'bejar'

if __name__ == '__main__':
    datapath = '../../Data/'

    X = np.loadtxt(datapath+'peaks2.csv', delimiter=' ')
    print(X.shape)
    # X = resample(X, X.shape[1]//2, axis=1, window=X.shape[1])
    # print(X.shape)

    # fig = plt.figure(figsize=(12,6))
    # plt.plot(data[5])
    # plt.plot(data[12])
    # plt.show()
    # dist, cost, acc_cost, path = dtw(data[5], data[12], dist=distance.euclidean)
    # print(dist)
    # plt.imshow(cost.T, origin='lower', interpolation='nearest')
    # plt.plot(path[0], path[1], 'w')
    # plt.xlim((-0.5, cost.shape[0]-0.5))
    # plt.ylim((-0.5, cost.shape[1]-0.5))
    # plt.show()

    mdist = np.zeros(X.shape[0] * (X.shape[0] - 1) // 2)
    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            mdist[sel(X.shape[0], i, j)] = fastdtw(X[i], X[j], dist=distance.euclidean)[0]
    nc = 3

    km = KMedoidsFlexible(n_clusters=nc, distance='precomputed')
    labels1 = km.fit_predict(mdist)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for i in km.cluster_medoids_:
    #     plt.plot(X[i])
    # plt.show()

    mdist = np.zeros(X.shape[0] * (X.shape[0] - 1) // 2)
    for i in range(X.shape[0]):
        for j in range(i+1, X.shape[0]):
            mdist[sel(X.shape[0], i, j)] = distance.euclidean(X[i], X[j])

    km = KMedoidsFlexible(n_clusters=nc, distance='precomputed')
    labels2 = km.fit_predict(mdist)

    fig = plt.figure()

    for i in range(nc):
        ax = fig.add_subplot(2, nc, i + 1)
        for p, j in enumerate(labels1):
            if j == i:
                ax.plot(X[p])

        ax = fig.add_subplot(2,nc,nc+i+1)
        for p, j in enumerate(labels2):
            if j == i:
              ax.plot(X[p])
    plt.show()
