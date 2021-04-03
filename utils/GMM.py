# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    # 屏蔽开始print
    # 更新W
        # self.weights = np.zeros(())
    # 更新pi
        self.pi = [ 1.0/n_clusters] * n_clusters
    # 更新Mu
        self.Mu = np.random.random((n_clusters,2))
    # 更新Var
        self.Var = np.empty((n_clusters, 2, 2))
        for i in range(n_clusters):
            self.Var[i] = np.eye(2) * np.random.rand(1) * n_clusters
    # 屏蔽结束

    def fit(self, data):
        # 作业3
        # 屏蔽开始
        iter = 0
        self.posterior = np.zeros((data.shape[0], self.n_clusters))
        while iter < self.max_iter :
            iter = iter + 1
            density = np.zeros((self.posterior.shape))
            #1 . update weight with u, sigma, pi
            for j in range(self.n_clusters):
                # posterior_J = np.array([ multivariate_normal.pdf( data[i, :], self.Mu[j,:], self.Var[j,:,:] ) * self.pi[j] for i in range(data.shape[0])])
                # self.posterior[:, j] = posterior_J
                norm = multivariate_normal(self.Mu[j], self.Var[j])
                density[:, j] = norm.pdf(data)
            self.posterior = density * self.pi

            self.posterior = self.posterior / self.posterior.sum(axis=1, keepdims=True)

            #2. update mu, sigma , pi
            self.nk = np.sum(self.posterior, 0)

            self.Mu = np.tensordot( self.posterior, data, [0, 0]) / self.nk.reshape(-1,1)
            for j in range(self.n_clusters):
                tmp = data - self.Mu[j, :]
                self.Var[j, :, :] = np.dot(tmp.T * self.posterior[:, j], tmp) / self.nk[j]

            self.pi = self.nk / data.shape[0]
            # print(self.Var)
            # print(self.Mu)

        self.gaussian = []
        for j in range(self.n_clusters):
            gau =  multivariate_normal(self.Mu[j,:], self.Var[j,:,:])
            self.gaussian.append(gau)

        # 屏蔽结束

    def predict(self, data):
        # 屏蔽开始
        results = []
        for i in range(data.shape[0]):
            prob = []
            for j in range(self.n_clusters):
                density = self.gaussian[j].pdf(data[i, :])
                prob.append(density * self.pi[j])

            # import pdb; pdb.set_trace()
            results.append(np.argsort(prob)[-1])
        return results
        # 屏蔽结束

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化



