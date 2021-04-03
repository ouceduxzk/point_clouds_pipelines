# 文件功能： 实现 K-Means 算法

import numpy as np
from copy import deepcopy
class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        iter = 0
        labels = np.random.choice(self.k_, data.shape[0])

        new_centers = np.zeros((self.k_, data.shape[1]))
        #idxs = np.random.choice(data.shape[0], self.k_)
        self.centers = data[np.arange(self.k_), :]
        #self.centers = np.zeros((self.k_, data.shape[1]))
        # for i in range(self.k_):
        #     import pdb; pdb.set_trace()
        #     self.centers[i,:] = np.mean(data[np.where(labels == i), :], 1)

        while iter < self.max_iter_ :
            # assign label to each point
            for i in range(data.shape[0]):
                point = data[i, :]
                norms = np.sum(np.abs(point - self.centers)** 2, axis = -1)
                labels[i] = norms.argsort()[0]
            # update center
            for i in range(self.k_):
                ids = np.where(labels == i)
                new_centers[i, :] = np.mean(data[ids, :], axis = 1)
            #print(new_centers)

            # err = np.sum(np.abs(new_centers - self.centers)**2)
            # if err < self.tolerance_:
            #     break
            optimized = True
            for i in range(self.k_):
                if np.linalg.norm(new_centers[i] - self.centers[i])> self.tolerance_:
                    optimized = False

            if optimized :
                break
            iter = iter + 1

            self.centers = deepcopy(new_centers) # need deepcopy ?
        # 屏蔽结束
        #print(self.centers)
    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始
        for i in range(p_datas.shape[0]):
            point = p_datas[i, :]
            norms = np.sum(np.abs(point - self.centers)** 2, axis = -1)
            pred = norms.argsort()[0]
            result.append(pred)
        # 屏蔽结束
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    k_means = K_Means(n_clusters=2)
    k_means.fit(x)

    cat = k_means.predict(x)
    print(cat)

