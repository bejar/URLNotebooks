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
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
from numpy.random import normal
import numpy as np
from dtw import dtw
from kemlglearn.cluster import KMedoidsFlexible
from fastdtw import fastdtw

__author__ = 'bejar'

if __name__ == '__main__':
    datapath = '../../Data/'

    data = np.loadtxt(datapath+'peaks.csv', delimiter=' ')
    print(data.shape)
    fig = plt.figure(figsize=(12,6))
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

    mdist = np.zeros((data.shape[0], data.shape[0]))
    for i in range(data.shape[0]):
        print(i)
        for j in range(i+1, data.shape[0]):
            mdist[i,j] = fastdtw(data[i], data[j], dist=distance.euclidean)[0]
            mdist[j,i] = mdist[i,j]

    plt.imshow(mdist, origin='lower', interpolation='nearest')
    plt.show()
