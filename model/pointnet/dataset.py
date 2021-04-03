import torch
import torch.utils.data as data
import numpy as np
import os, sys

class ModelNetDataset(data.Dataset):
  def __init__(self, root, split = "train",  num_points = 2500, aug = True):
    self.root = root
    self.category_file = os.path.join(self.root, "modelnet40_shape_names.txt")
    self.input_file = os.path.join(self.root, "modelnet40_{}.txt".format(split))
    self.num_points = num_points
    self.aug = aug
    self.category  = {}
    self.fns = []

    with open(self.category_file, "r") as fp :
      for i, line in enumerate(fp):
        obj = line.strip()
        self.category[obj] = i

    self.classes = list(self.category.keys())
    with open(self.input_file, "r") as fp :
      for line in fp:
        #cate = line.strip().split("_")[0]
        self.fns.append(line.strip())


  def __getitem__(self, index):
    fn = self.fns[index]
    category = fn.split("_")[0]
    cls_name = self.category[category]
    print(cls_name)

    file_path = os.path.join(self.root, category , fn + ".npy")
    print(file_path)
    data = np.load(file_path)
    sample_idx = np.random.choice(data.shape[0], self.num_points, replace=False)
    samples = data[smaple_idx, :3] #xyz only

    # the data is already normalized , no need to do that for now

    # augmentaiton
    #if self.aug:

    point_samples = torch.from_numpy(samples.astype(np.float32))
    point_cls = torch.from_numpy(np.array([cls_name]).astype(np.int64))
    return point_samples, point_cls

  def __len__(self):
    return len(self.fns)