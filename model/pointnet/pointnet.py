import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class TNet(nn.Module):
  def __init__(self):
    super(TNet, self).__init__()
    self.conv1 = nn.Conv1d(3, 64, 1)
    self.conv2 = nn.Conv1d(64, 128, 1)
    self.conv3 = nn.Conv1d(128, 1024, 1)

    self.fc1 = nn.Linear(1024, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 9)

    self.bn1 = nn.BatchNorm1d(64)
    self.bn2 = nn.BatchNorm1d(128)
    self.bn3 = nn.BatchNorm1d(1024)

    self.bn4 = nn.BatchNorm1d(512)
    self.bn5 = nn.BatchNorm1d(256)

  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x)))

    x = torch.max(x, 2 , keepdim=True)[0]
    x = x.view(-1, 1024)
    x = F.relu(self.bn4(self.fc1(x)))
    x = F.relu(self.bn5(self.fc2(x)))
    x = self.fc3(x)

    #not sure why we need to add a identity matrix
    batch_size = x.size()[0]
    iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1, 9).repeat(batch_size,1)
    if x.is_cuda :
      iden = iden.cuda()

    x = x + iden
    x = x.view(-1, 3, 3)
    return x

class PointNetFeature(nn.Module):
  def __init__(self, global_features = True):
    super(PointNetFeature, self).__init__()
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
    N = x.size()[2]
    trans = self.input_transform(x)
    x = x.transpose(2, 1)
    # n x 3 with 3 x 3
    x = torch.bmm(x, trans)
    x = x.transpose(2, 1)
    x = F.relu(self.bn1(self.conv1(trans)))
    #skip feature transform for now

    local_feat = x
    x = F.relu(self.bn2(self.conv2(x)))
    x = self.bn3(self.conv3(x))
    x = torch.max(x, 2, keepdim=True)[0]
    x = x.view(-1, 1024)

    if self.global_features :
      return x, trans
    else:
      x = x.view(-1, 1024, 1).repeat(1,1, N)
      return torch.cat([x, local_feat], 1), trans


class PointNetCls(nn.Module):
  def __init__(self, feature_transform, num_classes):
    super(PointNetCls, self).__init__()
    self.num_classes = num_classes
    self.feat, trans = PointNetFeature()
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
    return F.log_softmax(x, dim=1), trans




