import os
import cv2
import torch
import pickle
import numpy as np

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.nms_wrapper import nms
from faster_rcnn.deploy import im_detect, test_net

from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file, get_output_dir

# hyper-parameters
# ------------
imdb_name = 'coco_2014_minival'
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
trained_model = 'checkpoints/coco-vgg16.hdf5'

rand_seed = 1024

save_name = 'coco_pretrained'
max_per_image = 300
thresh = 0.05
vis = False

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)


if __name__ == '__main__':
    # load data
    imdb = get_imdb(imdb_name)
    imdb.competition_mode(on=True)

    # load net
    net = FasterRCNN(classes=imdb.classes, debug=True)
    network.load_net(trained_model, net)
    print('load model successfully!')

    net.cuda()
    net.eval()

    # evaluation
    test_net(save_name, net, imdb, max_per_image, thresh=thresh, vis=vis)
