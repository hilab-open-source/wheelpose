import contextlib
import copy
import io
import itertools
import json
import logging
import os
import numpy as np
import scipy as sp
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluator, DatasetEvaluators

KPT_OKS_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

COCO_PERSON_KEYPOINT_NAMES = (
     "nose",
     "left_eye", "right_eye",
     "left_ear", "right_ear",
     "left_shoulder", "right_shoulder",
     "left_elbow", "right_elbow",
     "left_wrist", "right_wrist",
     "left_hip", "right_hip",
     "left_knee", "right_knee",
     "left_ankle", "right_ankle",
)

class KeypointsEvaluator(COCOEvaluator):
    def reset(self):
        self._predictions=[]

    def evaluate(self, img_ids=None):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[KeypointsEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "instances" in predictions[0]:
            self._eval_keypoint_predictions(predictions, img_ids=img_ids)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_keypoint_predictions(self, predictions, img_ids=None):
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )

        coco_eval = (
            get_keypoint_metrics(
                self._coco_api,
                coco_results,
            )
            if len(coco_results) > 0
            else None  # cocoapi does not handle empty results very well
        )

        if coco_eval is None:
            return
        else:
            eucl, kpt_eucl, overall_odj, kpt_odj, pjpe, mpjpe, ub_mpjpe, lb_mpjpe, arms_mpjpe, torso_mpjpe, head_mpjpe, legs_mpjpe, wrists_mpjpe, elbows_mpjpe, shoulders_mpjpe, hips_mpjpe, knees_mpjpe, ankles_mpjpe, no_ankles_mpjpe= coco_eval
            self._results["overall_pdj"] = overall_odj
            self._results["oks_individual_keypoints_euclidean"] = eucl
            self._results["oks_individual_keypoints_euclidean_kpt_scaled"] = kpt_eucl
            self._results["individual_keypoints_pdj"] = kpt_odj
            self._results["pjpe"] = pjpe
            self._results["mpjpe"] = mpjpe
            self._results["upper_body_mpjpe"] = ub_mpjpe
            self._results["lower_body_mpjpe"] = lb_mpjpe
            self._results["arms_mpjpe"] = arms_mpjpe
            self._results["torso_mpjpe"] = torso_mpjpe
            self._results["legs_mpjpe"] = legs_mpjpe
            self._results["head_mpjpe"] = head_mpjpe
            self._results["wrists_mpjpe"] = wrists_mpjpe
            self._results["elbows_mpjpe"] = elbows_mpjpe
            self._results["shoulders_mpjpe"] = shoulders_mpjpe
            self._results["hips_mpjpe"] = hips_mpjpe
            self._results["knees_mpjpe"] = knees_mpjpe
            self._results["ankles_mpjpe"] = ankles_mpjpe
            self._results["no_ankles_mpjpe"] = no_ankles_mpjpe
            

def get_keypoint_metrics(
    coco_gt,
    coco_results,
    # iou_type,
    # kpt_oks_sigmas=None,
    # cocoeval_fn=COCOeval_opt,
    img_ids=None,
    # max_dets_per_image=None,
):
    coco_dt = coco_gt.loadRes(coco_results)

    gts = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[]))
    dts = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=[]))
    inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
    dts = [dts[i] for i in inds]

    vars = (KPT_OKS_SIGMAS * 2)**2
    k = len(vars)
    
    pjpe_scores = [] #Per joint position error
    mpjpe_scores = [] #mean pjpe
    upper_body_mpjpe_scores = [] #[0-10]
    lower_body_mpjpe_scores = [] #[11-16]
    arms_mpjpe_scores = [] #[7-10] elbows and wrist
    torso_mpjpe_scores = [] #[5,6,11,12] shoulder and hips
    head_mpjpe_scores = [] #[0-4]
    legs_mpjpe_scores = [] #[13-16] knee and ankle
    wrists_mpjpe_scores = [] #[9-10]
    elbows_mpjpe_scores = [] #[7-8]
    shoulders_mpjpe_scores = [] #[5-6]
    hips_mpjpe_scores = [] #[11-12]
    knees_mpjpe_scores = [] #[13-14]
    ankles_mpjpe_scores = [] #[15-16]
    no_ankles_mpjpe_scores = [] #[0-14]

    scaled_euclideans = []
    scaled_kpt_euclideans = []

    kp_pdjs = []
    overall_pdjs = []

    ious = np.zeros(len(dts), len(gts))
    for j, gt in enumerate(gts):
        # create bounds for ignore regions(double the gt bbox)
        g = np.array(gt['keypoints'])
        xg = g[0::3]; yg = g[1::3]; vg = g[2::3]
        k1 = np.count_nonzero(vg > 0)
        bb = gt['bbox']
        x0 = bb[0] - bb[2]; x1 = bb[0] + bb[2] * 2
        y0 = bb[1] - bb[3]; y1 = bb[1] + bb[3] * 2
        diagonal = np.sqrt(bb[2]**2 + bb[3]**2)
        bb = gt["bbox"]

        temp_pjpe_sums = [] #summed euclidean pjpe disance
        temp_pjpe_scores = [] #Per joint position error

        for i, dt in enumerate(dts):
            d = np.array(dt['keypoints'])
            xd = d[0::3]; yd = d[1::3]

            if k1>0:
                # measure the per-keypoint distance if keypoints visible
                dx = xd - xg
                dy = yd - yg

            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                z = np.zeros(k)
                dx = np.max((z, x0-xd),axis=0)+np.max((z, xd-x1),axis=0)
                dy = np.max((z, y0-yd),axis=0)+np.max((z, yd-y1),axis=0)

            #reused values
            dx_dy_sqd = dx**2+dy**2
            euclidean_d = dx_dy_sqd**0.5

            kp_pdj = euclidean_d < (.05*diagonal)

            overall_pdj = np.sum(kp_pdj) / k1
            if overall_pdj != 0: # removes some of the completely false detections
                overall_pdjs.append(overall_pdj)
                kp_pdjs.append(np.where(vg > 0, kp_pdj, np.nan))

            e = dx_dy_sqd / (gt['area']+np.spacing(1)) / 2 # oks without the per keypoint factor
            e_kpt = e / vars
            e=np.where(vg > 0, e, np.nan)
            
            #reused val
            exp_neg_e = np.exp(-e)

            #oks
            ious[i, j] = np.nanmean(exp_neg_e)

            e_kpt=np.where(vg > 0, e_kpt, np.nan)
            scaled_euclideans.append(exp_neg_e)
            scaled_kpt_euclideans.append(np.exp(-e_kpt))

            temp_pjpe_scores.append(euclidean_d)
            temp_pjpe_sums.append(sum(euclidean_d))

        #calculate min distance to current gt and select it
        min_sum = min(temp_pjpe_sums)
        min_index = temp_pjpe_sums.index(min_sum)
        min_pjpe_array = temp_pjpe_scores[min_index]

        pjpe_scores.append(min_pjpe_array)
        mpjpe_scores.append(np.nanmean(min_pjpe_array))
        upper_body_mpjpe_scores.append(np.nanmean(min_pjpe_array[:11]))
        lower_body_mpjpe_scores.append(np.nanmean(min_pjpe_array[11:]))
        arms_mpjpe_scores.append(np.nanmean(min_pjpe_array[7:11]))
        torso_mpjpe_scores.append(np.nanmean(np.array(min_pjpe_array)[[5,6,11,12]]))
        head_mpjpe_scores.append(np.nanmean(min_pjpe_array[:5]))
        legs_mpjpe_scores.append(np.nanmean(min_pjpe_array[13:]))
        wrists_mpjpe_scores.append(np.nanmean(min_pjpe_array[9:11]))
        elbows_mpjpe_scores.append(np.nanmean(min_pjpe_array[7:9]))
        shoulders_mpjpe_scores.append(np.nanmean(min_pjpe_array[5:7]))
        hips_mpjpe_scores.append(np.nanmean(min_pjpe_array[11:13]))
        knees_mpjpe_scores.append(np.nanmean(min_pjpe_array[13:15]))
        ankles_mpjpe_scores.append(np.nanmean(min_pjpe_array[15:]))
        no_ankles_mpjpe_scores.append(np.nanmean(min_pjpe_array[:15]))

    #oks calculations
    #oks_50
    print(ious)
    1/0
    #iou_bool_50 = ious >= 0.5
    #tp_50 = np.sum(iou_bool_50)
    #fp_50 = np.sum(~iou_bool_50)

    kp_pdjs = np.nanmean(np.stack(kp_pdjs), axis=0)
    kp_pdjs_dict = {COCO_PERSON_KEYPOINT_NAMES[i]:kp_pdjs[i] for i in range(k)}

    scaled_euclideans = np.nanmean(np.stack(scaled_euclideans), axis=0)
    scaled_euclidean_dict = {COCO_PERSON_KEYPOINT_NAMES[i]:scaled_euclideans[i] for i in range(k)}

    scaled_kpt_euclideans = np.nanmean(np.stack(scaled_kpt_euclideans), axis=0)
    scaled_kpt_euclidean_dict = {COCO_PERSON_KEYPOINT_NAMES[i]:scaled_kpt_euclideans[i] for i in range(k)}

    pjpe_scores = np.nanmean(np.stack(pjpe_scores), axis=0)
    pjpe_dict = {COCO_PERSON_KEYPOINT_NAMES[i]:pjpe_scores[i] for i in range(k)} #new

    overall_pdj_dict = {"overall_pdj": np.mean(overall_pdjs)}

    mpjpe_dict = {"mpjpe": np.mean(mpjpe_scores)} #new

    upper_body_mpjpe_dict = {"upper_body_mpjpe": np.mean(upper_body_mpjpe_scores)} #new
    lower_body_mpjpe_dict = {"lower_body_mpjpe": np.mean(lower_body_mpjpe_scores)} #new
    arms_mpjpe_dict = {"arms_mpjpe": np.mean(arms_mpjpe_scores)} #new
    torso_mpjpe_dict = {"torso_mpjpe": np.mean(torso_mpjpe_scores)} #new
    head_mpjpe_dict = {"head_mpjpe": np.mean(head_mpjpe_scores)} #new
    legs_mpjpe_dict = {"legs_mpjpe": np.mean(legs_mpjpe_scores)} #new
    wrists_mpjpe_dict = {"wrists_mpjpe": np.mean(wrists_mpjpe_scores)} #new
    elbows_mpjpe_dict = {"elbows_mpjpe": np.mean(elbows_mpjpe_scores)} #new
    shoulders_mpjpe_dict = {"shoulders_mpjpe": np.mean(shoulders_mpjpe_scores)} #new
    hips_mpjpe_dict = {"hips_mpjpe": np.mean(hips_mpjpe_scores)} #new
    knees_mpjpe_dict = {"knees_mpjpe": np.mean(knees_mpjpe_scores)} #new
    ankles_mpjpe_dict = {"ankles_mpjpe": np.mean(ankles_mpjpe_scores)} #new
    no_ankles_mpjpe_dict = {"no_ankles_mpjpe": np.mean(no_ankles_mpjpe_scores)} #new

    return scaled_euclidean_dict, scaled_kpt_euclidean_dict, overall_pdj_dict, kp_pdjs_dict, pjpe_dict, mpjpe_dict, upper_body_mpjpe_dict, lower_body_mpjpe_dict, arms_mpjpe_dict, torso_mpjpe_dict, head_mpjpe_dict, legs_mpjpe_dict, wrists_mpjpe_dict, elbows_mpjpe_dict, shoulders_mpjpe_dict, hips_mpjpe_dict, knees_mpjpe_dict, ankles_mpjpe_dict, no_ankles_mpjpe_dict
