"""
Code for training custom vg detection...
"""
import cv2
import os
import torch
import numpy as np
from datetime import datetime

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer

from faster_rcnn.roi_data_layer.layer import RoIDataLayer
from faster_rcnn.datasets.vg_hdf5 import VisualGenome
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
from argparse import ArgumentParser
try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is None or color is None:
        print(text, flush=True)
    else:
        cprint(text, color=color, on_color=on_color, attrs=attrs)

parser = ArgumentParser(description='Train script')
parser.add_argument('-emb', type=str, help='What to use for embeddings')
parser.add_argument('-n', default=500000, type=int, help='Num steps')
parser.add_argument('-lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('-eps', default=1e-8, type=float, help='epsilon')
parser.add_argument('-skip', default=10, type=int, help='layers to skip')
args = vars(parser.parse_args())

# hyper-parameters
# ------------
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
pretrained_model = 'checkpoints/coco-vgg16.hdf5'
output_dir = 'checkpoints/visual-genome2'

start_step = 0
end_step = args['n']

rand_seed = 123456
_DEBUG = True
use_tensorboard = False
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard
# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY*5
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load data
imdb = VisualGenome(split=0)
roidb = imdb.roidb
data_layer = RoIDataLayer(roidb, imdb.num_classes)

# load net
net = FasterRCNN(classes=imdb.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.01)
network.load_net(pretrained_model, net)
net.cuda()
net.train()

params = list(net.parameters())

print("Params are {}".format(
    '\n'.join(['{}: {} {}'.format(n, p.size(), '(opt)' if i < args['skip']*2 else '     ')
               for i, (n,p) in enumerate(net.named_parameters())])
))
optimizer = torch.optim.Adam(params[args['skip']*2:], lr=args['lr'], weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                             eps=args['eps'])

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        exp_name = datetime.now().strftime('vgg16_%m-%d_%H-%M')
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

# training
train_loss = 0
tp, tf, fg, bg = 0., 0., 0, 0
step_cnt = 0
re_cnt = False
t = Timer()
t.tic()
for step in range(start_step, end_step+1):

    # get one batch
    blobs = data_layer.forward()
    ######
    im_data = blobs['data']
    im_info = blobs['im_info']
    gt_boxes = blobs['gt_boxes']
    gt_ishard = blobs['gt_ishard']
    dontcare_areas = blobs['dontcare_areas']

    # forward
    net(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
    loss = net.loss + net.rpn.loss

    if _DEBUG:
        tp += float(net.tp)
        tf += float(net.tf)
        fg += net.fg_cnt
        bg += net.bg_cnt

    train_loss += loss.data[0]
    step_cnt += 1

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration

        log_text = 'step %d, image: %s, loss: %.4f, fps: %.2f (%.2fs per batch)' % (
            step, blobs['im_name'], train_loss / step_cnt, fps, 1./fps)
        log_print(log_text, color='green', attrs=['bold'])

        if _DEBUG:
            log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt))
            log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                net.rpn.cross_entropy.data.cpu().numpy()[0], net.rpn.loss_box.data.cpu().numpy()[0],
                net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
            )
        re_cnt = True

    if use_tensorboard and step % log_interval == 0:
        exp.add_scalar_value('train_loss', train_loss / step_cnt, step=step)
        exp.add_scalar_value('learning_rate', args['lr'], step=step)
        if _DEBUG:
            exp.add_scalar_value('true_positive', tp/fg*100., step=step)
            exp.add_scalar_value('true_negative', tf/bg*100., step=step)
            losses = {'rpn_cls': float(net.rpn.cross_entropy.data.cpu().numpy()[0]),
                      'rpn_box': float(net.rpn.loss_box.data.cpu().numpy()[0]),
                      'rcnn_cls': float(net.cross_entropy.data.cpu().numpy()[0]),
                      'rcnn_box': float(net.loss_box.data.cpu().numpy()[0])}
            exp.add_scalar_dict(losses, step=step)

    if (step % 20000 == 0) and step > 0:
        save_name = os.path.join(output_dir, 'faster_rcnn_{}.h5'.format(step))
        network.save_net(save_name, net)
        print(('save model: {}'.format(save_name)))

    if re_cnt:
        tp, tf, fg, bg = 0., 0., 0, 0
        train_loss = 0
        step_cnt = 0
        t.tic()
        re_cnt = False

