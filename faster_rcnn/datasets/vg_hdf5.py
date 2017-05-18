import os
from faster_rcnn.datasets.imdb import imdb
import numpy as np
import copy
import scipy.sparse
import h5py, json
from faster_rcnn.fast_rcnn.config import cfg
import pickle
from faster_rcnn.datasets.vg_detection_eval import VGeval
from collections import defaultdict

class VisualGenome(imdb):
    def __init__(self, roidb_file='VG-SGG.h5', dict_file='VG-SGG-dicts.json',
                 imdb_file='imdb_1024.h5', split=0, num_im=-1):
        imdb.__init__(self, roidb_file[:-3])

        self.vg_path = os.path.join(cfg.DATA_DIR, 'visual_genome')

        # read in dataset from a h5 file and a dict (json) file
        self.im_h5 = h5py.File(os.path.join(self.vg_path, imdb_file), 'r')
        self.roi_h5 = h5py.File(os.path.join(self.vg_path, roidb_file), 'r')

        # roidb metadata
        self.info = json.load(open(os.path.join(self.vg_path,
                                                dict_file), 'r'))
        self.im_refs = self.im_h5['images'] # image data reference
        im_scale = self.im_refs.shape[2]

        print(('split==%i' % split))
        data_split = self.roi_h5['split'][:]

        self.split = split
        if split >= 0:
            split_mask = data_split == split # current split
        else: # -1
            split_mask = data_split >= 0 # all

        # get rid of images that do not have box
        valid_mask = self.roi_h5['img_to_first_box'][:] > 0
        valid_mask = np.bitwise_and(split_mask, valid_mask)
        self._image_index = np.where(valid_mask)[0] # split index

        if num_im > -1:
            self._image_index = self._image_index[:num_im]

        # override split mask
        split_mask = np.zeros_like(data_split).astype(bool)
        split_mask[self.image_index] = True  # build a split mask
        # if use all images
        self.im_sizes = np.vstack([self.im_h5['image_widths'][split_mask],
                                   self.im_h5['image_heights'][split_mask]]).transpose()

        # h5 file is in 1-based index
        self.im_to_first_box = self.roi_h5['img_to_first_box'][split_mask]
        self.im_to_last_box = self.roi_h5['img_to_last_box'][split_mask]
        self.all_boxes = self.roi_h5['boxes_%i' % im_scale][:]  # will index later
        self.all_boxes[:, :2] = self.all_boxes[:, :2]
        assert(np.all(self.all_boxes[:, :2] >= 0))  # sanity check
        assert(np.all(self.all_boxes[:, 2:] > 0))  # no empty box


        # convert from xc, yc, w, h to x1, y1, x2, y2
        self.all_boxes[:, :2] = self.all_boxes[:, :2] - self.all_boxes[:, 2:]/2
        self.all_boxes[:, 2:] = self.all_boxes[:, :2] + self.all_boxes[:, 2:]
        self.labels = self.roi_h5['labels'][:,0]

        # add background class
        self.info['label_to_idx']['__background__'] = 0
        self.class_to_ind = self.info['label_to_idx']
        self._classes = sorted(self.class_to_ind, key=lambda k: self.class_to_ind[k])


        # load relation labels
        self.im_to_first_rel = self.roi_h5['img_to_first_rel'][split_mask]
        self.im_to_last_rel = self.roi_h5['img_to_last_rel'][split_mask]
        self._relations = self.roi_h5['relationships'][:]
        self._relation_predicates = self.roi_h5['predicates'][:,0]
        assert(self.im_to_first_rel.shape[0] == self.im_to_last_rel.shape[0])
        assert(self._relations.shape[0] == self._relation_predicates.shape[0]) # sanity check
        self.predicate_to_ind = self.info['predicate_to_idx']
        self.predicate_to_ind['__background__'] = 0
        self.ind_to_predicates = sorted(self.predicate_to_ind, key=lambda k:
                                  self.predicate_to_ind[k])

        # Default to roidb handler
        self.set_proposal_method('gt')

        # For now....
        self.competition_mode(False)

    def im_getter(self, idx):
        w, h = self.im_sizes[idx, :]
        ridx = self.image_index[idx]
        im = self.im_refs[ridx]
        im = im[:, :h, :w] # crop out
        im = im.transpose((1,2,0)) # c h w -> h w c
        return im

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        """
        gt_roidb = []
        for i in range(self.num_images):
            assert(self.im_to_first_box[i] >= 0)
            boxes = self.all_boxes[self.im_to_first_box[i]
                                   :self.im_to_last_box[i]+1,:]

            gt_classes = self.labels[self.im_to_first_box[i]
                                     :self.im_to_last_box[i]+1]

            overlaps = np.zeros((boxes.shape[0], self.num_classes))
            for j, o in enumerate(overlaps): # to one-hot
                #if gt_classes[j] > 0: # consider negative sample
                o[gt_classes[j]] = 1.0
            max_overlaps = overlaps.max(axis=1)
            max_classes = overlaps.argmax(axis=1)
            overlaps = scipy.sparse.csr_matrix(overlaps)

            # make ground-truth relations
            gt_relations = []
            if self.im_to_first_rel[i] >= 0: # if image has relations
                predicates = self._relation_predicates[self.im_to_first_rel[i]
                                             :self.im_to_last_rel[i]+1]
                obj_idx = self._relations[self.im_to_first_rel[i]
                                             :self.im_to_last_rel[i]+1]
                obj_idx = obj_idx - self.im_to_first_box[i]
                assert(np.all(obj_idx>=0) and np.all(obj_idx<boxes.shape[0])) # sanity check
                for j, p in enumerate(predicates):
                    gt_relations.append([obj_idx[j][0], obj_idx[j][1], p])

            gt_relations = np.array(gt_relations)

            seg_areas = np.multiply((boxes[:, 2] - boxes[:, 0] + 1),
                                    (boxes[:, 3] - boxes[:, 1] + 1)) # box areas
            gt_roidb.append({
                'boxes': boxes,
                'gt_classes' : gt_classes,
                'gt_overlaps' : overlaps,
                'gt_relations': gt_relations,
                'flipped' : False,
                'seg_areas' : seg_areas,
                'db_idx': i,
                'image': lambda im_i=i: self.im_getter(im_i),
                'roi_scores': np.ones(boxes.shape[0]),
                'width': self.im_sizes[i][0],
                'height': self.im_sizes[i][1],
                'max_classes': max_classes,
                'max_overlaps': max_overlaps,
            })
        return gt_roidb

    def _vg_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_index):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
                [{'image_id': int(index),
                  'category_id': int(cat_id),
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_vg_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind, self.num_classes - 1))
            results.extend(self._vg_results_one_category(all_boxes[cls_ind],
                                                           cls_ind))
        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
        print('{:.1f}'.format(100 * ap_default))
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            print('{: <14}: {:.1f}'.format(cls, 100 * ap))

        print('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = os.path.join(output_dir, 'vg_results')
        res_file += '.json'
        self._write_vg_results_file(all_boxes, res_file)

        gts = defaultdict(list)
        preds = defaultdict(list)
        for category_id, boxes in enumerate(all_boxes):
            for img_id, b in enumerate(boxes):
                for row in b:
                    preds[(img_id, category_id)].append((row[-1], row[:4]))

        for i in range(self.num_images):
            for lab, box in zip(
                    self.labels[self.im_to_first_box[i]:self.im_to_last_box[i]+1],
                    self.all_boxes[self.im_to_first_box[i]:self.im_to_last_box[i]+1],
            ):
                gts[i, lab].append(box)

        vg_eval = VGeval(gts, preds)
        vg_eval.evaluate()
        vg_eval.accumulate()
        self._print_detection_eval_metrics(vg_eval)
        eval_file = os.path.join(output_dir, 'vg_detection_eval.pkl')
        with open(eval_file, 'wb') as fid:
            pickle.dump(vg_eval, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote VG eval results to: {}'.format(eval_file))
        self.vg_eval = vg_eval

    def _get_widths(self):
        return self.im_sizes[:,0]
