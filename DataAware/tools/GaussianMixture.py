import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture,BayesianGaussianMixture
import pickle

def GM(x, gm_file='/root/resnet20/tmp/gm.pkl',n_components=10,train=True):
    print('GaussianMixture start...')
    if train:
        print('use train data')
        #gmm = BayesianGaussianMixture(n_components=n_components, tol=1e-3,max_iter=100,covariance_type='full').fit(x) 
        gmm = GaussianMixture(n_components=n_components,covariance_type='full').fit(x)
        with open(gm_file, 'wb') as pickle_file:
            pickle.dump(gmm, pickle_file)

    with open('/root/resnet20/tmp/gm.pkl', 'rb') as pickle_file:
        gmm = pickle.load(pickle_file)

    y_pred = gmm.predict(x)
    print('GaussianMixture done.')
    return y_pred
    plt.scatter(x[:, 0], x[:, 1], c=y_pred)
    plt.show()
