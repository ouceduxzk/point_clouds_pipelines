from model.pointnet import dataset
from model.pointnet import pointnet

from absl import app
from absl import flags


FLAGS = flags.FLAGS
flags.DEFINE_string("input", "", "folder for input")


def main(argv):
  dataloader = dataset.ModelNetDataset(FLAGS.input)



if __name__ == '__main__':
    app.run(main)