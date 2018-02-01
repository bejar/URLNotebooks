"""
.. module:: Consensustest

Consensustest
*************

:Description: Consensustest

    

:Authors: bejar
    

:Version: 

:Created on: 31/01/2018 10:57 

"""

from __future__ import print_function, division
from scipy.spatial import distance
from scipy.signal import resample
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

__author__ = 'bejar'

# part = [[0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2],
#         [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
#         [0, 0, 0, 0, 1, 0, 1, 1, 1, 2, 2, 2]]
#
# cooc = np.zeros((12, 12), dtype=int)
#
# for p in part:
#     for i in range(12):
#         for j in range(12):
#             if p[i] == p[j]:
#                 cooc[i,j] +=1
#
# print(cooc)
# sn.clustermap(cooc, annot=True, fmt="d", cmap="mako", col_cluster=False, row_colors='rrrbbbbbbggg')
# plt.show()


h = np.array([[1,0,0,1,0,1,0],[1,0,0,1,0,1,0], [1,0,1,0,0,0,1],[0,1,1,0,0,0,1],[0,1,0,0,1,0,1]])


print(np.dot(h,h.T)*(1/3))