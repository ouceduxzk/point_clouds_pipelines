import numpy as np

from KMeans import K_Means
from sklearn.cluster import KMeans
from scipy.spatial import distance
from sklearn.metrics import pairwise_distances
import scipy

class SpectralCluster(object):
    # k是分组数
    def __init__(self, n_clusters=2):
        self.k_ = n_clusters
        self.kmeans =  K_Means(n_clusters=n_clusters)

    def squared_exponential(self, x, y, sig=0.8, sig2=1):
        norm = np.linalg.norm(x - y)
        dist = norm * norm
        return np.exp(- dist / (2 * sig * sig2))

    def affinity(self, data):
        N = data.shape[0]
        sig = []
        ans = np.zeros((N, N))
        for i in range(N):
          dists = []
          for j in range(N):
            dis = np.linalg.norm(data[i, :] - data[j,:])
            dists.append(dis)
          dists.sort()
          sig.append(np.mean(dists[:5]))

        for i in range(N):
          for j in range(N):
            ans[i][j] = self.squared_exponential(data[i], data[j], sig[i], sig[j])
        return ans

    def affinity_fast(self, data):
        N = data.shape[0]
        sig = []
        ans = np.zeros((N, N))
        dists = distance.cdist(data, data)

        dists.sort()
        sig = np.mean(dists[:, :5], axis =1) # neighour of 5 distances as variance

        for i in range(N):
          for j in range(N):
            ans[i][j] = self.squared_exponential(data[i], data[j], sig[i], sig[j])

        return ans

    def get_laplacian_features(self, data):
        N = data.shape[0]
        W = self.affinity_fast(data)
        D_half_inv = np.zeros(W.shape)
        tmp = np.sum(W, axis=1)
        D_half_inv.flat[::len(tmp) + 1] = tmp ** (-0.5)
        #import pdb; pdb.set_trace()
        L = D_half_inv.dot(W).dot(D_half_inv)  #graph laplacian

        w, v = scipy.sparse.linalg.eigs(L, self.k_)
        X = v.real
        rows_norm = np.linalg.norm(X, axis=1, ord=2)
        X = (X.T /rows_norm).T
        return X

    def fit(self, data):
        V = self.get_laplacian_features(data)
        self.kmeans.fit(V)

    def predict(self, data):
        V = self.get_laplacian_features(data)
        return self.kmeans.predict(V)

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    cluster = SpectralCluster(n_clusters=2)
    cluster.fit(x)

    cat = cluster.predict(x)
    print(cat)

