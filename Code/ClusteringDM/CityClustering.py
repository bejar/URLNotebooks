"""
.. module:: CityClustering

CityClustering
*************

:Description: CityClustering

    

:Authors: bejar
    

:Version: 

:Created on: 30/09/2015 14:07 

"""

__author__ = 'bejar'

from sklearn.cluster import Birch, MiniBatchKMeans, KMeans
import numpy as np
from numpy import loadtxt
import time
from sklearn.metrics import adjusted_mutual_info_score
import warnings
warnings.filterwarnings("ignore")

citypath = '/home/bejar/Data/City/'

# Data from the City dataset (BCNpos, PARpos, LONpos)
data = 'LONcrime.csv'

citypos = loadtxt(citypath+data, delimiter=',')

print(data, citypos.shape)

# KMeans
km = KMeans(n_clusters=100, n_init=1)
itime = time.perf_counter()
kmlabels = km.fit_predict(citypos)
etime = time.perf_counter()
print ('K-means Time = ', etime-itime)

# Minibatch Kmeans
itime = time.perf_counter()
mbkm = MiniBatchKMeans(n_clusters=100, batch_size=1000, n_init=1, max_iter=5000)
mbkmlabels = mbkm.fit_predict(citypos)
etime = time.perf_counter()
print ('MB K-means Time = ', etime-itime)

print('Similarity Km vs MBKm', adjusted_mutual_info_score(kmlabels, mbkmlabels))

# Birch
itime = time.perf_counter()
birch = Birch(threshold=0.02, n_clusters=100, branching_factor=100)
birchlabels = birch.fit_predict(citypos)
etime = time.perf_counter()
print ('BIRCH Time = ',etime-itime)

print('Similarity Km vs BIRCH',adjusted_mutual_info_score(kmlabels, birchlabels))



