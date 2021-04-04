from model.pointnet import dataset
from model.pointnet import pointnet
import torch
import torch.nn.functional as F
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
  classifier  = pointnet.PointNetCls(40)
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
      points, labels = points.cuda(), labels.cuda()
      optimizer.zero_grad()
      classifier = classifier.train()
      pred, trans = classifier(points)
      loss = F.nll_loss(pred, labels)
      loss.backward()
      optimizer.step()
      pred_label = pred.data.max(1)[1]
      correct = pred_label.eq(labels.data).cpu().sum()
      if i % 20 ==0 :
        print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(FLAGS.batch_size)))

  torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % ("model_saved", epoch))

  total_correct = 0
  total_testset = 0
  for j, data in enumerate(test_dataloader, 0):
    points, labels = data
    labels = labels[:, 0]
    points = points.transpose(2, 1)
    points, labels = points.cuda(), labels.cuda()
    classifier = classifier.eval()
    pred, _ = classifier(points)
    pred_label = pred.data.max(1)[1]
    correct = pred_label.eq(labels.data).cpu().sum()
    total_correct += correct.item()
    total_testset += points.size()[0]

  print("final accuracy {}".format(total_correct / float(total_testset)))


if __name__ == '__main__':
    app.run(main)