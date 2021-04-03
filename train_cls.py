from model.pointnet import dataset
from model.pointnet import pointnet
import torch

from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_string("input", "", "folder for input")
flags.DEFINE_integer("workers", 4 ,"# of workers")
flags.DEFINE_integer("num_points", 2500, "# of points")
flags.DEFINE_integer("batch_size", 8, "batch size")
flags.DEFINE_integer("epochs", 1, "# of epochs")
flags.DEFINE_string("model", "", "load pretrained model")

def main(argv):
  # setup dataloader

  train_dataset = dataset.ModelNetDataset(FLAGS.input, "train", FLAGS.num_points)
  test_dataset = dataset.ModelNetDataset(FLAGS.input, "test", FLAGS.num_points)

  train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size = FLAGS.batch_size, shuffle=True,
    num_workers = FLAGS.workers
  )

  test_dataloader =  torch.utils.data.DataLoader(
    test_dataset, batch_size = 1, shuffle=False,
    num_workers = FLAGS.workers
  )

  # setup classifier
  classifier  = pointnet.PointNetCls(True, 40)
  if FLAGS.model != "":
    classifier.load_state_dict(torch.load(FLAGS.model))
  optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
  classifier.cuda()

  num_batch = len(train_dataset) / FLAGS.batch_size
  for epoch in range(FLAGS.epochs):
    scheduler.step()
    for i, data in enumerate(train_dataloader, 0):
      points, labels = data
      labels = labels[:, 0]
      points = points.transpose(2, 1) # bxnx3 -> bx3xn
      points, labels = point.cuda(), labels.cuda()
      optimizer.zero_grad()
      classifier = classifier.train()
      pred = classifier(points)




if __name__ == '__main__':
    app.run(main)