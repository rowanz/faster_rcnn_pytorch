"""
Modified from pycocotools
"""

import numpy as np
import datetime
import time
from collections import defaultdict
import copy
from faster_rcnn.pycocotools import mask


class VGeval:
    # Interface for evaluating detection on the Microsoft COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = CocoEval(cocoGt,cocoDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  useSegm    - [1] if true evaluate against ground-truth segments
    #  useCats    - [1] if true use category labels for evaluation    # Note: if useSegm=0 the evaluation is run on bounding boxes.
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    def __init__(self, gts, dts):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param gts: Defaultdict (img_id, class_id) -> [bounding boxes]
        :param dts: Defaultdict (img_id, class_id) -> [(score, bbox)]
        :param cocoDt: coco object with detection results
        :return: None
        '''
        self.gts = gts
        self.dts = dts

        self.params = {}  # evaluation parameters
        self.evalImgs = defaultdict(list)  # per-image per-category evaluation results
        # [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self.params = Params()  # parameters
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts

        img_ids, classes = zip(*gts.keys())
        self.params.imgIds = list(np.unique(img_ids))
        self.params.catIds = list(np.unique(classes))
        self.params.maxDets = sorted(self.params.maxDets)


    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        print('Running per image evaluation...      ')

        tic = time.time()
        # loop through images, area range, max detection number
        self.ious = {(imgId, catId): self.computeIoU(imgId, catId, self.params.maxDets[-1])
                     for imgId in self.params.imgIds
                     for catId in self.params.catIds}

        self.evalImgs = [self.evaluateImg(imgId, catId, areaRng, self.params.maxDets[-1])
                         for catId in self.params.catIds
                         for areaRng in self.params.areaRng
                         for imgId in self.params.imgIds
                         ]
        toc = time.time()
        print('DONE (t=%0.2fs).' % (toc - tic))

    def computeIoU(self, imgId, catId, max_dets):
        gt = self.gts[imgId, catId]
        dt = self.dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return []
        dt = sorted(dt, key=lambda x: -x[0])
        if len(dt) > max_dets:
            dt = dt[0:max_dets]

        # compute iou between each dt and gt region
        is_crowd = [0 for x in gt] # WTF is this
        return mask.iou([d[1] for d in dt], gt, is_crowd)

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        #
        p = self.params
        gt = self.gts[imgId, catId]
        dt = self.dts[imgId, catId]
        if len(gt) == 0 and len(dt) == 0:
            return None

        # Highest score first
        dt = sorted(dt, key=lambda d: -d[1])[:maxDet]
        # load computed ious
        N_iou = len(self.ious[imgId, catId])
        ious = self.ious[imgId, catId][0:maxDet, np.arange(len(gt))] if N_iou > 0 else self.ious[
            imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        dtIg = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iouThrs):
                for dind, (sc, d) in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0:
                            continue
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # match successful and best so far, store appropriately
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dtm[tind, dind] = m
                    gtm[tind, m] = dind

        # store results for given image and category
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [x for x in range(len(dt))],
            'gtIds': [x for x in range(len(gt))],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d[0] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def accumulate(self):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...   ')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        p = self.params
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds)
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones((T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))

        # create dictionary for future indexing
        catIds = p.catIds
        setK = set(catIds)
        setA = set(map(tuple, p.areaRng))
        setM = set(p.maxDets)
        setI = set(p.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate([tuple(x) for x in p.areaRng]) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        # K0 = len(p.catIds)
        I0 = len(p.imgIds)
        A0 = len(p.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [_f for _f in E if _f]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')

                    dtm = np.concatenate([e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    dtIg = np.concatenate([e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:, inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = len([ig for ig in gtIg if ig == 0])
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs)
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                        except:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'precision': precision,
            'recall': recall,
        }
        toc = time.time()
        print('DONE (t=%0.2fs).' % (toc - tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6} | maxDets={:>3} ] = {}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '%0.2f:%0.2f' % (
            p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else '%0.2f' % (iouThr)
            areaStr = areaRng
            maxDetsStr = '%d' % (maxDets)

            aind = [i for i, aRng in enumerate(['all', 'small', 'medium', 'large']) if
                    aRng == areaRng]
            mind = [i for i, mDet in enumerate([1, 10, 100]) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                # areaRng
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            print(iStr.format(titleStr, typeStr, iouStr, areaStr, maxDetsStr,
                              '%.3f' % (float(mean_s))))
            return mean_s

        if not self.eval:
            raise Exception('Please run accumulate() first')
        self.stats = np.zeros((12,))
        self.stats[0] = _summarize(1)
        self.stats[1] = _summarize(1, iouThr=.5)
        self.stats[2] = _summarize(1, iouThr=.75)
        self.stats[3] = _summarize(1, areaRng='small')
        self.stats[4] = _summarize(1, areaRng='medium')
        self.stats[5] = _summarize(1, areaRng='large')
        self.stats[6] = _summarize(0, maxDets=1)
        self.stats[7] = _summarize(0, maxDets=10)
        self.stats[8] = _summarize(0, maxDets=100)
        self.stats[9] = _summarize(0, areaRng='small')
        self.stats[10] = _summarize(0, areaRng='medium')
        self.stats[11] = _summarize(0, areaRng='large')

    def __str__(self):
        self.summarize()


class Params:
    '''
    Params for coco evaluation api
    '''

    def __init__(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, np.round((1.00 - .0) / .01) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2],
                        [96 ** 2, 1e5 ** 2]]
