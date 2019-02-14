"""
.. module:: ValidationAuthors

PartitionalAuthors
*************

:Description: ValidationAuthors

    Validation clustering measures for the authors dataset

:Authors: bejar
    

:Version: 

:Created on: 18/09/2015 9:58 

"""


from os import listdir
from os.path import join
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from kemlglearn.metrics.cluster import calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import argparse
import numpy as np

__author__ = 'bejar'

path = '../../Data/authors/'

def authors_data(method=1, nfeatures=100):
    """
    Returns a data matrix representing the documents using the indicated method for generating the features and the
    specified number of features and the labels for the documents

    :param method:
    :param nfeatures:
    :return:
    """

    docs = sorted(listdir(docpath))[1:]

    # use v[:2] for dataset labels, use [2:-2] for individual authors
    labs = [v[:2] for v in docs]
    ulabs = sorted(list(set(labs)))
    dlabs = {}
    for ul, v  in zip(ulabs, range(len(ulabs))):
        dlabs[ul]=v
    labels = [dlabs[v] for v in labs]

    pdocs = [join(docpath, f) for f in docs]

    if method == 1: # Features are word counts
        cvec = CountVectorizer(input='filename', stop_words='english', max_features=nfeatures)
    elif method == 2: # Features are TF-IDF
        cvec = TfidfVectorizer(input='filename', stop_words='english', max_features=nfeatures)
    elif method == 3: # Features are word occurence
        cvec = TfidfVectorizer(input='filename', stop_words='english', max_features=nfeatures, binary=True, use_idf=False, norm=False)

    return cvec.fit_transform(pdocs).toarray(), np.array(labels).reshape(1, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=1, choices=[1, 2], help='Dataset to use (1 or 2)', type=int)
    parser.add_argument('--nfeatures', help="Number of features to generate", default=100, type=int)
    parser.add_argument('--fmethod', help="Method for generating features", default=1, choices=[1, 2, 3], type=int)
    parser.add_argument('--nclusters', help="Number of clusters", default=10, type=int)
    args = parser.parse_args()

    if args.data == 1:
        docpath = path + 'Auth1/'
    else:
        docpath = path + 'Auth2/'

    nfeatures = args.nfeatures
    method = args.fmethod

    nclusters = args.nclusters

    authors, alabels = authors_data(method, nfeatures)

    pca = PCA()
    fdata = pca.fit_transform(authors)

    # KMeans

    lscores = []
    for nc in range(2,nclusters+1):
        km = KMeans(n_clusters=nc, n_init=10, random_state=0)
        labels = km.fit_predict(authors)
        lscores.append((
            silhouette_score(authors, labels),
            calinski_harabasz_score(authors, labels),
            davies_bouldin_score(authors, labels)))

    fig = plt.figure()
    fig.suptitle('K-means', fontsize=20)
    ax = fig.add_subplot(131)
    plt.title('Sil')
    plt.plot(range(2,nclusters+1), [x for x,_,_ in lscores])
    ax = fig.add_subplot(132)
    plt.title('CH')
    plt.plot(range(2,nclusters+1), [x for _, x,_ in lscores])
    ax = fig.add_subplot(133)
    plt.title('DB')
    plt.plot(range(2,nclusters+1), [x for _, _, x in lscores])

    plt.show()


    # GMM
    # covariance_type = 'spherical', 'tied', 'diag', 'full'

    lscores = []
    for nc in range(2,nclusters+1):
        gmm = GaussianMixture(n_components=nc, covariance_type='full', random_state=0)
        gmm.fit(authors)
        labels = gmm.predict(authors)
        lscores.append((
            silhouette_score(authors, labels),
            calinski_harabasz_score(authors, labels),
            davies_bouldin_score(authors, labels)))

    fig = plt.figure()
    fig.suptitle('GMM', fontsize=20)
    ax = fig.add_subplot(131)
    plt.title('Sil')
    plt.plot(range(2,nclusters+1), [x for x,_,_ in lscores])
    ax = fig.add_subplot(132)
    plt.title('CH')
    plt.plot(range(2,nclusters+1), [x for _, x,_ in lscores])
    ax = fig.add_subplot(133)
    plt.title('DB')
    plt.plot(range(2,nclusters+1), [x for _, _, x in lscores])

    plt.show()

    # Spectral Clustering
    lscores = []
    for nc in range(2,nclusters+1):
        spec = SpectralClustering(n_clusters=nc, affinity='nearest_neighbors', n_neighbors=15, random_state=0)
        labels = spec.fit_predict(authors)
        lscores.append((
            silhouette_score(authors, labels),
            calinski_harabasz_score(authors, labels),
            davies_bouldin_score(authors, labels)))

    fig = plt.figure()
    fig.suptitle('Spectral', fontsize=20)
    ax = fig.add_subplot(131)
    plt.title('Sil')
    plt.plot(range(2,nclusters+1), [x for x,_,_ in lscores])
    ax = fig.add_subplot(132)
    plt.title('CH')
    plt.plot(range(2,nclusters+1), [x for _, x,_ in lscores])
    ax = fig.add_subplot(133)
    plt.title('DB')
    plt.plot(range(2,nclusters+1), [x for _, _, x in lscores])

    plt.show()

    # print(adjusted_mutual_info_score(alabels, labels))
    #
    # show_figure(fdata, alabels, labels, '')


