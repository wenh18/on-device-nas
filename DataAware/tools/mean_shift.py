# -*- coding: utf-8 -*- 

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs


def mean_shift(x):
    print('mean_shift start...')

    bandwidth = estimate_bandwidth(x, quantile=0.003,n_samples = 5000, n_jobs=-1)
    print('bandwidth:{}'.format(bandwidth))
    bandwidth = 10
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(x)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    import matplotlib.pyplot as plt
    from itertools import cycle

    # plt.figure(1)
    # plt.clf()
    # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    # for k, col in zip(range(n_clusters_), colors):
    #     my_members = labels == k
    #     cluster_center = cluster_centers[k]
    #     plt.plot(x[my_members, 0], x[my_members, 1], col + '.')
    #     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #          markeredgecolor='k', markersize=14)
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()
    print('mean_shift done...')
    return n_clusters_, labels