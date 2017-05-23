import cv2
import numpy as np

import os


from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.deploy import test_net_sg
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file, get_output_dir
import pickle

# hyper-parameters
# ------------
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
trained_model = 'checkpoints/visual-genome/faster_rcnn_100000.h5'

rand_seed = 1024

save_name = 'vg'
max_per_image = 300
thresh = 0.05
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)

# load data
imdb = get_imdb('vg_val_small')
imdb.competition_mode(on=True)

# load net
net = FasterRCNN(classes=imdb.classes, debug=True)
network.load_net(trained_model, net)
print('load model successfully!')

net.cuda()
net.eval()

# # evaluation
# test_net(save_name, net, imdb, max_per_image, thresh=thresh, test_bbox_reg=cfg.TEST.BBOX_REG,
         # vis=False)
test_net_sg('blah', net, imdb)

# output_dir = get_output_dir(imdb, save_name)
# det_file = os.path.join(output_dir, 'detections.pkl')
#
# with open(det_file, 'rb') as f:
#     all_boxes = pickle.load(f)
#
# imdb.evaluate_detections(all_boxes, output_dir)
