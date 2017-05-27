import cv2
import numpy as np

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.deploy import test_net_sg, test_net, get_preds

from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file, get_output_dir

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

for imdb_fn in (
    'vg',
    'vg_val',
    ):
    imdb = get_imdb(imdb_fn)
    imdb.competition_mode(on=True)

    net = FasterRCNN(classes=imdb.classes, debug=True)
    network.load_net(trained_model, net)

    net.cuda()
    net.eval()
    print('load model successfully!')

    get_preds(imdb_fn + '_out', net, imdb, max_per_image,
        thresh=thresh, test_bbox_reg=cfg.TEST.BBOX_REG, vis=False)

print("DONE!")