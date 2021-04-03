
import os, sys
import numpy as np

'''
representation of a plane with point P(x, x, z) and normal
'''
class Plane:
  def __init__(self, origin, normal):
    self.origin = origin
    self.normal = normal

  # compute the distance of a point to a plane
  def distance(self, point):
    return np.fabs(np.dot((point - self.origin), self.normal))


