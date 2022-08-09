import time
import argparse
import os
import sys
import torch
import cv2
import numpy as np
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results
from detectron2.structures import Instances
from detectron2.layers.nms import batched_nms
from detectron2.structures import Boxes, Instances
from soco_device import DeviceCheck

from frcnn_ext_models import add_config
#from frcnn_ext_models.bua.layers.nms import nms

from tqdm import tqdm
from typing import List, Generator, Sequence

import logging
from detectron2.config import CfgNode as CN

logging.getLogger("fvcore.common.checkpoint").setLevel(logging.ERROR)
logging.getLogger("detectron2.engine.defaults").setLevel(logging.ERROR)

PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
TEST_SCALES = (600,)
TEST_MAX_SIZE = 1000


class Pack(dict):
    def __getattr__(self, name):
        return self[name]

    def clone_dict(self, x):
        for k, v in list(x.items()):
            self[k] = v

    def add(self, **kwargs):
        for k, v in list(kwargs.items()):
            self[k] = v

    def copy(self):
        pack = Pack()
        for k, v in list(self.items()):
            if type(v) is list:
                pack[k] = list(v)
            else:
                pack[k] = v
        return pack

def chunks(l: Sequence, n: int = 5) -> Generator[Sequence, None, None]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def load_vocab_file(filename):
    objects = []
    with open(filename) as f:
        for i in f.readlines():
            objects.append(i.strip())
    id_objects_map = {i: v for i, v in enumerate(objects)}
    return id_objects_map


def switch_extract_mode(mode):
    if mode == 'roi_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 1]
    elif mode == 'bboxes':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 2]
    elif mode == 'bbox_feats':
        switch_cmd = ['MODEL.BUA.EXTRACTOR.MODE', 3, 'MODEL.PROPOSAL_GENERATOR.NAME', 'PrecomputedProposals']
    else:
        print('Wrong extract mode! ')
        exit()
    return switch_cmd


def set_min_max_boxes(min_max_boxes):
    if min_max_boxes == 'min_max_default':
        return []
    try:
        min_boxes = int(min_max_boxes.split(',')[0])
        max_boxes = int(min_max_boxes.split(',')[1])
    except:
        print('Illegal min-max boxes setting, using config default. ')
        return []
    cmd = ['MODEL.BUA.EXTRACTOR.MIN_BOXES', min_boxes,
           'MODEL.BUA.EXTRACTOR.MAX_BOXES', max_boxes]
    return cmd


def set_cuda_device():
    device_name, device_ids = DeviceCheck().get_device(n_gpu=1)
    device_name = '{}:{}'.format(device_name, device_ids[0]) if len(device_ids) == 1 else device_name
    cmd = ['MODEL.DEVICE', device_name]
    return cmd


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(switch_extract_mode(args.extract_mode))
    cfg.merge_from_list(set_min_max_boxes(args.min_max_boxes))
    cfg.merge_from_list(set_cuda_device())
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def get_image_blob(im, pixel_means):
    """Converts an image into a network input.
    Arguments:
        im (ndarray): a color image
    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    pixel_means = np.array([[pixel_means]])
    dataset_dict = {}
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= pixel_means

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    for target_size in TEST_SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > TEST_MAX_SIZE:
            im_scale = float(TEST_MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)

    dataset_dict["image"] = torch.from_numpy(im).permute(2, 0, 1)
    dataset_dict["im_scale"] = im_scale

    return dataset_dict


def normalize_box_feats(boxes, im_h, im_w):
    '''
    input: 10 * 1d torch array of len 4 (xmin, ymin, xmax, ymax); img height; img width
    output: np array with shape (num_boxes, 8)
    8: (xmin, ymin, xmax, ymax, xcent, ycent, wbox, hbox) normalized to -1,1
    '''
    # print(f'img width:{im_w} img height:{im_h}')
    # print(f'box 0: {boxes[0]}')
    # print(f'boxes: {boxes}')
    assert (torch.all(boxes[:, 0] <= im_w) and torch.all(boxes[:, 2] <= im_w))
    assert (torch.all(boxes[:, 1] <= im_h) and torch.all(boxes[:, 3] <= im_h))
    feats = torch.zeros((boxes.shape[0], 6))

    feats[:, 0] = boxes[:, 0] * 2.0 / im_w - 1  # xmin
    feats[:, 1] = boxes[:, 1] * 2.0 / im_h - 1  # ymin
    feats[:, 2] = boxes[:, 2] * 2.0 / im_w - 1  # xmax
    feats[:, 3] = boxes[:, 3] * 2.0 / im_h - 1  # ymax
    # feats[:, 4] = (feats[:, 0] + feats[:, 2]) / 2  # xcenter
    # feats[:, 5] = (feats[:, 1] + feats[:, 3]) / 2  # ycenter
    feats[:, 4] = feats[:, 2] - feats[:, 0]  # box width
    feats[:, 5] = feats[:, 3] - feats[:, 1]  # box height
    return feats


class FRCNNExtractor(object):
    def __init__(self, extractor_dir, mode='caffe', extract_mode='roi_feats', min_max_boxes='10,50'):
        args = {}
        args['config_file'] = os.path.join(extractor_dir, 'config.yaml')
        args['mode'] = mode
        args['extract_mode'] = extract_mode
        args['min_max_boxes'] = min_max_boxes
        args['eval_only'] = True
        args = Pack(args)
        cfg = setup(args)

        self.model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            os.path.join(extractor_dir, cfg.MODEL.WEIGHTS), resume=False
        )
        self.model.eval()
        self.cfg = cfg
        self.vg_objects = load_vocab_file(os.path.join(extractor_dir, 'objects_vocab.txt'))

    def fast_rcnn_inference_single_image(
            self,
            img,
            boxes,
            scores,
            features,
            dataset_dict,
            score_thresh: float,
            nms_thresh: float,
            topk_per_image: int,
    ):
        """
        Single-image inference. Return bounding-box detection results by thresholding
        on scores and applying non-maximum suppression (NMS).
        Args:
            Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
            per image.
        Returns:
            Same as `fast_rcnn_inference`, but for only one image.
        """
        valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]

        # scores = scores[:, :-1]
        num_bbox_reg_classes = boxes.shape[1] // 4
        # Convert to Boxes to use the `clip` function ...
        boxes = boxes / dataset_dict['im_scale']

        max_scores, max_classes = scores.max(1)

        # 2. Apply NMS for each class independently.
        keep = batched_nms(boxes, max_scores, max_classes, nms_thresh)
        if topk_per_image >= 0:
            keep = keep[:topk_per_image]
        # print('num boxes before: {} after: {}'.format(len(boxes), len(keep)))
        boxes, scores, features = boxes[keep], scores[keep], features[keep]
        image_objects = np.argmax(scores.numpy()[:, 1:], axis=1)

        image_h, image_w, _ = np.shape(img)
        loc_feat = normalize_box_feats(boxes, image_h, image_w)
        feat = torch.cat((features, loc_feat), axis=1)
        obj_label = [self.vg_objects[i] for i in image_objects]
        objects = ' '.join(obj_label)

        info = {
            'objects': objects,
            'img_feat': feat,
        }
        meta = {
            'obj_label': obj_label,
            'image_objects': image_objects,
            'bbox': boxes.tolist(),
            'image_h': image_h,
            'image_w': image_w,
        }

        return info, meta

    def post_process(self, cfg, im, dataset_dict, boxes, scores, feats, attr_scores=None):
        MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
        MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
        CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

        dets = boxes / dataset_dict['im_scale']
        # TODO: test if image_objects is really correct to write this way
        max_conf = torch.zeros((scores.shape[0])).to(scores.device)
        for cls_ind in range(1, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            #keep = nms(dets, cls_scores, 0.3)
            max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                         cls_scores[keep],
                                         max_conf[keep])

        keep_boxes = torch.nonzero(max_conf >= CONF_THRESH).flatten()
        if len(keep_boxes) < MIN_BOXES:
            keep_boxes = torch.argsort(max_conf, descending=True)[:MIN_BOXES]
        elif len(keep_boxes) > MAX_BOXES:
            keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]
        print('num boxes before: {} after: {}'.format(len(boxes), len(keep_boxes)))
        image_feat = feats[keep_boxes]
        image_bboxes = dets[keep_boxes]
        import pdb;
        pdb.set_trace()
        image_objects = np.argmax(scores[keep_boxes].numpy()[:, 1:], axis=1)

        image_h = np.size(im, 0)
        image_w = np.size(im, 1)
        loc_feat = normalize_box_feats(image_bboxes, image_h, image_w).to(image_feat.device)
        feat = torch.cat((image_feat, loc_feat), axis=1)

        objects = ' '.join([self.vg_objects[i] for i in image_objects])

        info = {
            'objects': objects,
            'img_feat': feat
        }

        return info

    def batch_extract_feat(self, imgs: List[np.ndarray], batch_size: int = 1):
        """
        extract rcnn feature in batch

        :param imgs: list of numpy array representing raw imgs to
        :type imgs: List[np.ndarray]
        :param batch_size: batch size in, defaults to 1
        :type batch_size: int, optional
        :return: list of torch tensor
        :rtype: List[torch.Tensor]
        """
        st = time.time()
        img_feat_list = []
        meta_list = []
        for b_imgs in tqdm(chunks(imgs, n=batch_size)):
            dataset_dicts = [get_image_blob(img, self.cfg.MODEL.PIXEL_MEAN) for img in b_imgs]
            # extract roi features
            attr_scores = None
            with torch.set_grad_enabled(False):
                if self.cfg.MODEL.BUA.ATTRIBUTE_ON:
                    boxes, scores, features_pooled, attr_scores = self.model(dataset_dicts)
                else:
                    boxes, scores, features_pooled = self.model(dataset_dicts)

            boxes = [box.tensor.cpu() for box in boxes]
            scores = [score.cpu() for score in scores]
            features_pooled = [feat.cpu() for feat in features_pooled]
            if attr_scores is not None:
                attr_scores = [attr_score.cpu() for attr_score in attr_scores]

            for img, data_dict, box, score, feat in zip(b_imgs, dataset_dicts, boxes, scores, features_pooled):
                # img_feat = self.post_process(self.cfg, img, data_dict, box, score, feat)
                img_feat, meta = self.fast_rcnn_inference_single_image(img, box, score, feat, data_dict,
                                                                       self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
                                                                       self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST, 50)
                img_feat_list.append(img_feat)
                meta_list.append(meta)
        times = time.time() - st
        #print('time for {} imgs: {} s'.format(len(imgs), times))
        #print('fps: {}'.format(len(imgs) / times))
        return img_feat_list, meta_list

    def image_feature_extraction(self, path):
        imgs = [cv2.imread(path)]
        results = self.batch_extract_feat(imgs, batch_size=1)
        return results

if __name__ == "__main__":
    extractor = FRCNNExtractor('resources/frcnn-bua-caffe-r101-with-attrs')
    #img_dir = '/home/vincent/proj/soco/soco/soco-image-sparta/data/coco/train2014'
    img_dir = '/home/quad-0/coco/val2014'
    img_path_list = os.listdir(img_dir)
    budget = 100
    imgs = []
    for i, img in enumerate(img_path_list[:3]):
        if "txt" in img:
            continue
        if i == budget:
            break
        imgs.append(cv2.imread(os.path.join(img_dir, img)))
    results = extractor.batch_extract_feat(imgs, batch_size=1)
    print (results)
