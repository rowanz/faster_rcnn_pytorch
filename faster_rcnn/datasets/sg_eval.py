"""
Adapted from Danfei Xu
"""
import numpy as np

with open('data/visual_genome/VG/predicate_list.txt','r') as f:
    ALL_CLASSES = ['bg'] + f.read().splitlines()
c_to_i = {c:i for i,c in enumerate(ALL_CLASSES)}

PRIORS = {
    'on': .99,
    'has': .97,
    'in': .98,
    'of': .96,
    'wearing': .98,
    'near': .96,
    'with': .88,
    'above': .79,
    'holding': .8,
    'behind': .92,
    'under': .527,
    'sitting on': .50,
    'standing on': .61,
    'in front of': .59,
    'attached to': .29,
    'at': .70,
    'for': .31,
    'riding': .89,
}

class SceneGraphEvaluator:

    def __init__(self, imdb, mode):
        self.roidb = imdb.roidb
        self.result_dict = {}
        self.mode = mode

        self.result_dict = {}
        self.result_dict[self.mode + '_recall'] = {20:[], 50:[], 100:[]}


    def evaluate_scene_graph_entry(self, sg_entry, im_idx, iou_thresh):
        pred_triplets, triplet_boxes = \
            eval_detection_recall(sg_entry, self.roidb[im_idx],
                                  self.result_dict,
                                  self.mode,
                                  iou_thresh=iou_thresh)
        return pred_triplets, triplet_boxes


    def save(self, fn):
        np.save(fn, self.result_dict)


    def print_stats(self):
        print('======================' + self.mode + '============================')
        for k, v in self.result_dict[self.mode + '_recall'].items():
            print('R@%i: %f' % (k, np.mean(v)))



def eval_relation_recall(sg_entry,
                         roidb_entry,
                         result_dict,
                         mode,
                         iou_thresh):
    """
    For evaluating model on the relation tasks
    :param sg_entry: Predicted scene graph entry, dictionary-like object containing
                     boxes
                     relations (num_boxes, num_boxes, num_labels) -> scores
                     scores
    :param roidb_entry: Ground truth entry, dictionary like object containing
                     max_overlaps
                     boxes
                     gt_relations
                     gt_classes
    :param result_dict: Dictionary for storing results. TODO figure out what this does
    :param mode: pred_cls, sg_cls, or sg_det
    :param iou_thresh: 
    :return: Top k predicted triplets sorted by score, top k predicted boxes
    """
    # gt
    gt_inds = np.where(roidb_entry['max_overlaps'] == 1)[0]
    gt_boxes = roidb_entry['boxes'][gt_inds].copy().astype(float)
    num_gt_boxes = gt_boxes.shape[0]
    gt_relations = roidb_entry['gt_relations'].copy()
    gt_classes = roidb_entry['gt_classes'].copy()

    num_gt_relations = gt_relations.shape[0]
    if num_gt_relations == 0:
        return (None, None)
    gt_class_scores = np.ones(num_gt_boxes)
    gt_predicate_scores = np.ones(num_gt_relations)
    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_relations[:,2],
                                             gt_relations[:,:2],
                                             gt_classes,
                                             gt_boxes,
                                             gt_predicate_scores,
                                             gt_class_scores)

    # pred
    box_preds = sg_entry['boxes']
    num_boxes = box_preds.shape[0]
    predicate_preds = sg_entry['relations']
    class_preds = sg_entry['scores']
    predicate_preds = predicate_preds.reshape(num_boxes, num_boxes, -1)

    # no bg
    predicate_preds = predicate_preds[:, :, 1:]
    predicates = np.argmax(predicate_preds, 2).ravel() + 1
    predicate_scores = predicate_preds.max(axis=2).ravel()
    relations = []
    keep = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            if i != j:
                keep.append(num_boxes*i + j)
                relations.append([i, j])
    # take out self relations
    predicates = predicates[keep]
    predicate_scores = predicate_scores[keep]

    relations = np.array(relations)
    assert(relations.shape[0] == num_boxes * (num_boxes - 1))
    assert(predicates.shape[0] == relations.shape[0])
    num_relations = relations.shape[0]

    if mode =='pred_cls':
        # if predicate classification task
        # use ground truth bounding boxes
        assert(num_boxes == num_gt_boxes)
        classes = gt_classes
        class_scores = gt_class_scores
        boxes = gt_boxes
    elif mode =='sg_cls':
        assert(num_boxes == num_gt_boxes)
        # if scene graph classification task
        # use gt boxes, but predicted classes
        classes = np.argmax(class_preds, 1)
        class_scores = class_preds.max(axis=1)
        boxes = gt_boxes
    elif mode =='sg_det':
        # if scene graph detection task
        # use preicted boxes and predicted classes
        classes = np.argmax(class_preds, 1)
        class_scores = class_preds.max(axis=1)
        boxes = []
        for i, c in enumerate(classes):
            boxes.append(box_preds[i, c*4:(c+1)*4])
        boxes = np.vstack(boxes)
    else:
        raise NotImplementedError('Incorrect Mode! %s' % mode)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(predicates, relations, classes, boxes,
                 predicate_scores, class_scores)


    sorted_inds = np.argsort(relation_scores)[::-1]
    # compue recall
    for k in result_dict[mode + '_recall']:
        this_k = min(k, num_relations)
        keep_inds = sorted_inds[:this_k]
        recall = _relation_recall(gt_triplets,
                                  pred_triplets[keep_inds,:],
                                  gt_triplet_boxes,
                                  pred_triplet_boxes[keep_inds,:],
                                  iou_thresh)
        result_dict[mode + '_recall'][k].append(recall)

    # for visualization
    return pred_triplets[sorted_inds, :], pred_triplet_boxes[sorted_inds, :]


def _triplet(predicates, relations, classes, boxes,
             predicate_scores, class_scores):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-1) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-1), 2) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-1)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert(predicates.shape[0] == relations.shape[0])
    num_relations = relations.shape[0]
    triplets = np.zeros([num_relations, 3]).astype(np.int32)
    triplet_boxes = np.zeros([num_relations, 8]).astype(np.int32)
    triplet_scores = np.zeros([num_relations]).astype(np.float32)
    for i in range(num_relations):
        triplets[i, 1] = predicates[i]
        sub_i, obj_i = relations[i,:2]
        triplets[i, 0] = classes[sub_i]
        triplets[i, 2] = classes[obj_i]
        triplet_boxes[i, :4] = boxes[sub_i, :]
        triplet_boxes[i, 4:] = boxes[obj_i, :]
        # compute triplet score
        score =  class_scores[sub_i]
        score *= class_scores[obj_i]
        score *= predicate_scores[i]
        triplet_scores[i] = score
    return triplets, triplet_boxes, triplet_scores


def _relation_recall(gt_triplets, pred_triplets,
                     gt_boxes, pred_boxes, iou_thresh):

    # compute the R@K metric for a set of predicted triplets

    num_gt = gt_triplets.shape[0]
    num_correct_pred_gt = 0

    for gt, gt_box in zip(gt_triplets, gt_boxes):
        keep = np.zeros(pred_triplets.shape[0]).astype(bool)
        for i, pred in enumerate(pred_triplets):
            if gt[0] == pred[0] and gt[1] == pred[1] and gt[2] == pred[2]:
                keep[i] = True
        if not np.any(keep):
            continue
        boxes = pred_boxes[keep,:]
        sub_iou = iou(gt_box[:4], boxes[:,:4])
        obj_iou = iou(gt_box[4:], boxes[:,4:])
        inds = np.intersect1d(np.where(sub_iou >= iou_thresh)[0],
                              np.where(obj_iou >= iou_thresh)[0])
        if inds.size > 0:
            num_correct_pred_gt += 1
    return float(num_correct_pred_gt) / float(num_gt)


def iou(gt_box, pred_boxes):
    # computer Intersection-over-Union between two sets of boxes
    ixmin = np.maximum(gt_box[0], pred_boxes[:,0])
    iymin = np.maximum(gt_box[1], pred_boxes[:,1])
    ixmax = np.minimum(gt_box[2], pred_boxes[:,2])
    iymax = np.minimum(gt_box[3], pred_boxes[:,3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) +
            (pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) *
            (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps


################## New stuff

def eval_detection_recall(sg_entry, roidb_entry, result_dict, mode, iou_thresh=0.5):
    """
    Assuming we have an oracle that gets us the relations for us
    :param sg_entry: Predicted scene graph entry, dictionary-like object containing
                     boxes
                     scores
    :param roidb_entry: Ground truth entry, dictionary like object containing
                     max_overlaps
                     boxes
                     gt_relations - a (num_relations,3) array that is (box_1, box_2, relation)
                     gt_classes
    :param result_dict: Dictionary for storing results. TODO figure out what this does
    :param mode: pred_cls, sg_cls, or sg_det
    :param iou_thresh: 
    :return: Top k predicted triplets sorted by score, top k predicted boxes
    """
    # gt
    gt_inds = np.where(roidb_entry['max_overlaps'] == 1)[0]
    gt_boxes = roidb_entry['boxes'][gt_inds].copy().astype(float)
    num_gt_boxes = gt_boxes.shape[0]
    gt_relations = roidb_entry['gt_relations'].copy()
    gt_classes = roidb_entry['gt_classes'].copy()

    num_gt_relations = gt_relations.shape[0]
    if num_gt_relations == 0:
        return (None, None)
    gt_class_scores = np.ones(num_gt_boxes)
    gt_predicate_scores = np.ones(num_gt_relations)
    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_relations[:,2],
                                                gt_relations[:,:2],
                                                gt_classes,
                                                gt_boxes,
                                                gt_predicate_scores,
                                                gt_class_scores)
    class_preds = sg_entry['scores']
    # pred
    box_preds = sg_entry['boxes']
    num_classes = box_preds.shape[1] //4
    num_boxes = box_preds.shape[0]


    box_preds_nc = box_preds.reshape((num_boxes, num_classes, 4))
    box_preds_nc = box_preds_nc[np.arange(num_boxes), np.argmax(class_preds,1),:]


    # The oracle for the predictions ----------------------------------------------------
    ious = np.array([iou(gt_box, box_preds_nc) for gt_box in gt_boxes])
    print("IOU is of shape {}, whereas there are {} gt boxes and {} pred boxes".format(
        ious.shape, gt_boxes.shape, box_preds.shape
    ))
    predicate_preds = np.zeros((num_boxes, num_boxes, gt_relations[:,2].max()+1))+0.01
    for (gt_box_1, gt_box_2, pred) in gt_relations:
        box_1_overlaps = np.where(ious[gt_box_1] > iou_thresh)[0]
        box_2_overlaps = np.where(ious[gt_box_2] > iou_thresh)[0]
        # print("b1 {}, overlaps {}, b2 {}, overlaps {}".format(gt_box_1, box_1_overlaps,
        #                                                       gt_box_2,  box_2_overlaps))

        # Le oracle
        for b_1 in box_1_overlaps:
            for b_2 in box_2_overlaps:
                predicate_preds[b_1, b_2, pred] = 1.0
    #  ----------------------------------------------------

    # Will predict "On" if they overlap. ---------------------------------------------
    # ious = np.array([iou(box, box_preds_nc) for box in box_preds_nc])
    # print(np.where(ious>.3))
    # print("IOUs shape {}, gt preds {}".format(ious.shape, gt_relations[:,2]))
    # predicate_preds = np.zeros((num_boxes, num_boxes, len(ALL_CLASSES)))+0.01
    #
    # for b_1 in range(num_boxes):
    #     for b_2 in range(num_boxes):
    #         if b_1 != b_2 and ious[b_1, b_2] > 0.01 and ious[b_1, b_2] < .3:
    #             predicate_preds[b_1,b_2,:] = .5
    #             for pred, score in PRIORS.items():
    #                 predicate_preds[b_1, b_2, c_to_i[pred]] = score
    # -----------------------------------------------------------------------------

    # no bg
    predicate_preds = predicate_preds[:, :, 1:]
    predicates = np.argmax(predicate_preds, 2).ravel() + 1
    predicate_scores = predicate_preds.max(axis=2).ravel()
    relations = []
    keep = []
    for i in range(num_boxes):
        for j in range(num_boxes):
            if i != j:
                keep.append(num_boxes*i + j)
                relations.append([i, j])
    # take out self relations
    predicates = predicates[keep]
    predicate_scores = predicate_scores[keep]

    relations = np.array(relations)
    assert(relations.shape[0] == num_boxes * (num_boxes - 1))
    assert(predicates.shape[0] == relations.shape[0])
    num_relations = relations.shape[0]

    if mode =='pred_cls':
        # if predicate classification task
        # use ground truth bounding boxes
        assert(num_boxes == num_gt_boxes)
        classes = gt_classes
        class_scores = gt_class_scores
        boxes = gt_boxes
    elif mode =='sg_cls':
        assert(num_boxes == num_gt_boxes)
        # if scene graph classification task
        # use gt boxes, but predicted classes
        classes = np.argmax(class_preds, 1)
        class_scores = class_preds.max(axis=1)
        boxes = gt_boxes
    elif mode =='sg_det':
        # if scene graph detection task
        # use preicted boxes and predicted classes
        classes = np.argmax(class_preds, 1)
        class_scores = class_preds.max(axis=1)
        boxes = []
        for i, c in enumerate(classes):
            boxes.append(box_preds[i, c*4:(c+1)*4])
        boxes = np.vstack(boxes)
    else:
        raise NotImplementedError('Incorrect Mode! %s' % mode)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(predicates, relations, classes, boxes,
                 predicate_scores, class_scores)


    sorted_inds = np.argsort(relation_scores)[::-1]
    # compue recall
    for k in result_dict[mode + '_recall']:
        this_k = min(k, num_relations)
        keep_inds = sorted_inds[:this_k]
        recall = _relation_recall(gt_triplets,
                                  pred_triplets[keep_inds,:],
                                  gt_triplet_boxes,
                                  pred_triplet_boxes[keep_inds,:],
                                  iou_thresh)
        result_dict[mode + '_recall'][k].append(recall)

    # for visualization
    return pred_triplets[sorted_inds, :], pred_triplet_boxes[sorted_inds, :]
