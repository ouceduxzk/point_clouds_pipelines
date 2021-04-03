
import numpy as np
from plane import Plane

class RansacPlane:
  def __init__(self, options : dict, data : np.array):
    self.options = options
    self.data = data
    self.best_plane = None
    self.best_inliers = []
    self.best_num_inliers = 0
    self.N = self.data.shape[0]

  def fitPlane(self, sample):
    assert sample.shape[0] == 3
    p0 = sample[0, :]
    p1 = sample[1, :]
    p2 = sample[2, :]
    v1 = p1 - p0
    v2 = p1 - p2
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    return Plane(p0, normal)

  def randomIndices(self,):
    return np.random.choice(self.N, 3)

  def evaluate(self, plane):
    num_inliers = 0
    inliers = []
    for i in range(self.N):
      dis = plane.distance(self.data[i,:])
      if dis < self.options["max_error_distance"]:
        num_inliers = num_inliers + 1
        inliers.append(i)
    return num_inliers, inliers

  def get_inliers(self):
    return self.best_inliers

  def run(self):
    sample_indices = self.randomIndices()
    sample = self.data[sample_indices, :]
    plane = self.fitPlane(sample)

    num_inliers, inliers = self.evaluate(plane)
    self.best_num_inliers = num_inliers
    self.best_inliers = inliers

    iter = 0
    while iter < self.options["max_iter"] :
      iter = iter + 1
      sample_indices = self.randomIndices()
      sample = self.data[sample_indices, :]
      new_plane = self.fitPlane(sample)
      num_inliers, inliers = self.evaluate(new_plane)
      if num_inliers > self.best_num_inliers :
        self.best_num_inliers = num_inliers
        self.best_inliers = inliers

      if self.best_num_inliers > self.N*0.5:
        break

