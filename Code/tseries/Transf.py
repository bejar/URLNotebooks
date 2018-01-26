"""
.. module:: Transf

Transf
*************

:Description: Transf

    

:Authors: bejar
    

:Version: 

:Created on: 26/01/2018 12:59 

"""

from __future__ import print_function, division
from scipy.spatial import distance
from scipy.signal import resample
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
import numpy as np
from kemlglearn.cluster import KMedoidsFlexible
from fastdtw import fastdtw


__author__ = 'bejar'


if __name__ == '__main__':
    datapath = '../../Data/'

    X = np.loadtxt(datapath+'peaks2.csv', delimiter=' ')
    print(X.shape)
    X = resample(X, X.shape[1]//2, axis=1, window=X.shape[1])
    print(X.shape)
    #
    # np.savetxt(datapath+'peaks2.csv',X,delimiter=' ')
