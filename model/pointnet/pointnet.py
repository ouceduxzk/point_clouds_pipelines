import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class TNet(nn.Module):
  def __init__(self):
    pass
  def forward(self, x):
    pass


class PointNetFeature(nn.Module):
  def __init__(self, global_features : bool):
    self.global_features = global_features
    self.input_transform = TNet()
    self.feature_transform = TNet()

    self.conv1 = nn.Conv1d(3, 64, 1)
    self.conv2 = nn.Conv1d(64, 128, 1)
    self.conv3 = nn.Conv1d(128, 1024, 1)

    self.bn1 = nn.BatchNorm1d(64)
    self.bn2 = nn.BatchNorm1d(128)
    self.bn3 = nn.BatchNorm1d(1024)

  def forward(self, x):
    trans = self.input_transform(x)
    return x


class PointNetCls(nn.Module):
  def __init__(self, feature_transform, num_classes):
    self.num_classes = num_classes
    self.feat = PointNetFeature(global_feature = True)
    self.fc1 = nn.Linear(1024, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, self.num_classes)

    self.dropout = nn.Dropout(p = 0.25)
    self.bn1 = nn.BatchNorm1d(512)
    self.bn2 = nn.BatchNorm1d(256)

  def forward(self, x):
    x = self.feat(x)
    x = F.relu(self.bn1(self.fc1(x)))
    x = self.dropout(self.fc2(x))
    x = F.relu(self.bn2(x)())
    x = self.fc3(x)
    return F.log_softmax(x, dim=1)




