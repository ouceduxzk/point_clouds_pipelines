import glob
import os
import numpy as np
import open3d  as o3d
from open3d.geometry import KDTreeFlann as kdtree

def select_data(input ):
    outputs = []
    inputs = glob.glob(input + "/*")
    for category in inputs :
        if os.path.isdir(category):
            candidate = glob.glob(category + "/*.txt")[0]
            outputs.append(candidate)
    return outputs

def load_data(input):
    inputs = glob.glob(input + "/*")
    return inputs

def pca(data, sort = False):
    data = data.transpose()
    print(data.shape)
    data = data - np.mean(data, axis=1, keepdims=True)

    data_tp = data.transpose()

    H = np.matmul(data, data_tp)
    #svd
    eigen_vectors, eigen_values, _ = np.linalg.svd(H, full_matrices=True)
    if sort :
        sort_idxs = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[sort_idxs]
        eigen_vectors = eigen_vectors[:, sort_idxs]
    return eigen_vectors, eigen_values

def estimate_normal(data):
    neigh = kdtree.search_radius_vector_3d(data, 10)
    print(neigh)

if __name__ == '__main__':
    selected = load_data("./data/")

    all_projects = []
    for input_obj in selected:
        data = np.loadtxt(input_obj, delimiter=",")[:, :3]

        # pca
        eigen_vectors,eigen_values = pca(data)
        # project onto 2 main pc direction
        projected_data = np.dot(data, eigen_vectors[:, :2])
        projected_data = np.hstack([projected_data, np.zeros((data.shape[0], 1))])

        # projected_pc = o3d.geometry.PointCloud()
        # projected_pc.points = o3d.utility.Vector3dVector(projected_data)
        # all_projects.append(projected_pc)
        # # visualiztion
        # o3d.visualization.draw_geometries([projected_pc])


        estimate_normal(data)