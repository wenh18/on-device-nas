"""Find optimal number of clustres from a Dataset."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle

def k_means(x,k_means_file='/root/resnet20/tmp/k_means.pkl',n_clusters=10,train=True):
    if train:
        k_means = KMeans(n_clusters=n_clusters)
        k_means.fit(x)
        with open(k_means_file, 'wb') as pickle_file:
            pickle.dump(k_means, pickle_file)     

    with open(k_means_file, 'rb') as pickle_file:
        k_means = pickle.load(pickle_file)
        
    y_predict = k_means.predict(x)
    return y_predict
    # plt.scatter(x[:,0],x[:,1],c=y_predict)
    # plt.show()
    # print(k_means.predict((x[:30,:])))
    # print(metrics.calinski_harabasz_score(x,y_predict))
    # print(k_means.cluster_centers_)
    # print(k_means.inertia_)
    # print(metrics.silhouette_score(x,y_predict))