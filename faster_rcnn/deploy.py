# Code that visualizes detections, among other things
import cv2
import numpy as np
import os
import pickle

from faster_rcnn import network
from faster_rcnn.faster_rcnn import FasterRCNN, RPN
from faster_rcnn.utils.timer import Timer
from faster_rcnn.fast_rcnn.nms_wrapper import nms

from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file, get_output_dir
import matplotlib.cm as cmx
import matplotlib.colors as colors
from faster_rcnn.datasets.sg_eval import SceneGraphEvaluator


def get_cmap(N):
    """Returns a function that maps each index in 0, 1, ... N-1 to a distinct RGB color."""
    color_norm = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv')
    def map_index_to_rgb_color(index):
        pad = 40
        return np.round(np.array(scalar_map.to_rgba(index))*(255-pad)+pad)
    return map_index_to_rgb_color


def vis_detections(im, class_name, dets, thresh=0.05, color=(0, 204, 0)):
    """Visual debugging of detections."""
    # if dets.shape[0] > 0:
        # print("Class {}, dets shape {}, max scores {}".format(class_name, dets.shape, np.max(dets[:,-1])))

    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            # print("IM shape {}, bbox is {}, color is {}".format(im.shape, bbox, color))
            cv2.rectangle(im, bbox[0:2], bbox[2:4], color, thickness=3)

            txt = '{} {:.3f}'.format(class_name, score)
            (dx, dy), baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_PLAIN, 1, 3)

            # print("dx {} dy {} bl {}".format(dx, dy, baseline))
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0] + dx, bbox[1] - dy), color, thickness=-1)
            cv2.putText(im, txt, (bbox[0], bbox[1]),
                        cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness=1)
    return im


def im_detect(net, image, test_bbox_reg=True):
    """Detect object classes in an image given object proposals.
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    im_data, im_scales = net.get_image_blob(image)
    im_info = np.array(
        [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
        dtype=np.float32)

    cls_prob, bbox_pred, rois = net(im_data, im_info)
    scores = cls_prob.data.cpu().numpy()
    boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]

    if test_bbox_reg:
        #cfg.TEST.BBOX_REG
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy()
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, image.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


def test_net(name, net, imdb, max_per_image=300, thresh=0.05, test_bbox_reg=True, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    output_dir = get_output_dir(imdb, name)

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    cmap = get_cmap(imdb.num_classes)

    for i in range(num_images):

        im = imdb.im_getter(i)
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, test_bbox_reg)
        detect_time = _t['im_detect'].toc(average=False)

        _t['misc'].tic()
        if vis:
            # im2show = np.copy(im[:, :, (2, 1, 0)])
            im2show = np.copy(im, 'C')

        # skip j = 0, because it's the background class
        for j in range(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                im2show = vis_detections(im2show, imdb.classes[j], cls_dets, thresh=thresh,
                                         color=cmap(j))
            all_boxes[j][i] = cls_dets

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        nms_time = _t['misc'].toc(average=False)

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
            .format(i + 1, num_images, detect_time, nms_time))

        if vis:
            path = os.path.join(output_dir, 'viz')
            if not os.path.exists(path):
                os.mkdir(path)
            cv2.imwrite(os.path.join(path, '{}.jpg'.format(i)), im2show)

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)

def non_gt_rois(roidb):
    overlaps = roidb['max_overlaps']
    gt_inds = np.where(overlaps == 1)[0]
    non_gt_inds = np.setdiff1d(np.arange(overlaps.shape[0]), gt_inds)
    rois = roidb['boxes'][non_gt_inds]
    scores = roidb['roi_scores'][non_gt_inds]
    return rois, scores

def gt_rois(roidb):
    overlaps = roidb['max_overlaps']
    gt_inds = np.where(overlaps == 1)[0]
    rois = roidb['boxes'][gt_inds]
    return rois


def test_net_sg(name, net, imdb, mode='sg_det', max_per_image=100):
    """
    Tests the network the stanford scene-graph way.
    :param net_name: 
    :param weight_name: 
    :param imdb: 
    :param mode: 
    :param max_per_image: 
    :return: 
    """
    num_images = len(imdb.image_index)

    # timers
    _t = {'im_detect' : Timer(), 'evaluate' : Timer()}

    if mode == 'all':
        eval_modes = ['pred_cls', 'sg_cls', 'sg_det']
    else:
        eval_modes = [mode]

    # initialize evaluator for each task
    evaluators = {}
    for m in eval_modes:
        evaluators[m] = SceneGraphEvaluator(imdb, mode=m)

    for im_i in range(num_images):
        im = imdb.im_getter(im_i)

        for m in eval_modes:
            # if mode == 'pred_cls' or mode == 'sg_cls':
            #     use ground truth object locations
                # bbox_reg = False
                # box_proposals = gt_rois(roidb[im_i])
            _t['im_detect'].tic()

            scores, boxes = im_detect(net, im, test_bbox_reg=True)

            if boxes.size == 0:
                continue

            _t['im_detect'].toc()
            _t['evaluate'].tic()

            evaluators[m].evaluate_scene_graph_entry(
                {'boxes':boxes, 'scores':scores}, im_i, iou_thresh=0.5)
            _t['evaluate'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(im_i + 1, num_images, _t['im_detect'].average_time,
                      _t['evaluate'].average_time))

    # print out evaluation results
    for mode in eval_modes:
        evaluators[mode].print_stats()