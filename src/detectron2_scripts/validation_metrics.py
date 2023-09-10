# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
from pathlib import Path
import numpy as np
import pandas as pd
import os, json, cv2, random, shutil, argparse
import matplotlib.pyplot as plt

from keypoints_evaluator import KeypointsEvaluator

# import some common detectron2 utilities
from detectron2 import model_zoo

from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluator, DatasetEvaluators

def setup_cfg(weights_path=None, cfg_file="COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml", score_thresh=.9):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    if weights_path is not None:
        cfg.MODEL.WEIGHTS = str(weights_path.resolve())
    else:
        print("USING DETECTRON2 WEIGHTS")
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh

    return cfg

def evaluate(predictor, cfg, name, output_dir=None, keypoint=False):
    if output_dir is None:
        cocoevaluator = COCOEvaluator("real_test")
        keypointsevaluator = KeypointsEvaluator("real_test")
    else:
        cocoevaluator = COCOEvaluator("real_test", output_dir=output_dir)
        keypointsevaluator = KeypointsEvaluator("real_test", output_dir=output_dir)

    val_loader = build_detection_test_loader(cfg, name)
    
    if keypoint:
        return inference_on_dataset(predictor.model, val_loader, DatasetEvaluators([keypointsevaluator,]))
    else:
        return inference_on_dataset(predictor.model, val_loader, DatasetEvaluators([cocoevaluator,]))


def calc_single_trial_metrics(predictor, cfg, output_dir=None, score_thresh=.9, keypoint=False):
    if keypoint:
        output_dir
    print(output_dir)
    coco_eval = evaluate(predictor, cfg, "real_test", output_dir=str(output_dir), keypoint=keypoint)

    return coco_eval # overall metrics

def graph_ground_truth(visualizer, label, color="white"):
    bbox = label["bbox"]
    bbox_converted = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]

    visualizer.draw_box(bbox_converted, edge_color=color)
    
    kps = np.array(label["keypoints"]).reshape((-1,3))
    kps = kps[kps[:,2]!=0]

    for kp in kps:
        visualizer.draw_circle(kp[:2], color=color, radius=2)
    return visualizer

def graph_single_image_pred(im, predictor, dataset_metadata, gt_labels=None):
    output = predictor(im)
    v = Visualizer(im[:,:,::-1],
                    metadata=dataset_metadata,
                    # scale=.5,
                    # instance_mode=ColorMode.SEGMENTATION
                    )

    out = v.draw_instance_predictions(output["instances"].to("cpu"))
    if gt_labels is not None:
        for label in gt_labels:
            v = graph_ground_truth(v, label)

        out = v.get_output()

    return out.get_image()[:,:,::-1]

def graph_sample_images(predictor,
                        images_dir,
                        output_dir,
                        n_samples=-1,
                        gt_json_path=None,
                        dataset_metadata=MetadataCatalog.get(f"coco")):
    # getting all the ground truth labels
    if gt_json_path is not None:
        with open(gt_json_path, "r") as f:
            gt = json.load(f)
        n_images = len(gt["images"])

        if n_samples == -1:
            n_samples = n_images
        sample_idx = np.round(np.linspace(0, n_images-1, n_samples)).astype(int)
        gt_img_dict = {gt["images"][i]["id"]:{"file_path": images_dir / gt["images"][i]["file_name"], "anns":[]} for i in sample_idx}

        # ground truth annotations
        for ann in gt["annotations"]:
            curr_img_id = ann["image_id"]
            if curr_img_id not in gt_img_dict:
                continue
            gt_img_dict[curr_img_id]["anns"].append(ann)
        gt_img_dict = {gt_img["file_path"]: gt_img["anns"] for gt_img in gt_img_dict.values()}
    else:

        images = [images_dir.iterdir()]
        n_images = len(images)
        sample_idx = np.round(np.linspace(0, n_images-1, n_samples)).astype(int)
        gt_img_dict = {images[i]:None for i in sample_idx}


    # saving out single images
    for img_path, gt_labels in gt_img_dict.items():
        im = cv2.imread(str(img_path))
        annotated_im = graph_single_image_pred(im, predictor, dataset_metadata, gt_labels=gt_labels)
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), annotated_im)

    # im_path = list(images_dir.iterdir())[0]
    # im = cv2.imread(str(im_path))
    # graph_single_image_pred(im, predictor, dataset_metadata)

def process_evaluations(evaluations, output_dir, keypoint=False):
    if len(evaluations) == 0:
        return

    idx = list(evaluations.keys())

    split_evaluations = {}
    for evaluation in evaluations.values():
        for key, metrics in evaluation.items():
            if key not in split_evaluations:
                split_evaluations[key] = []
            split_evaluations[key].append(metrics)

    for annotation_type, metric in split_evaluations.items():
        metric_df = pd.DataFrame.from_records(metric)
        metric_df.set_index([pd.Index(idx)])

        plt.figure()
        plt.ylabel("Precision")
        plt.xlabel("Epoch")
        if not keypoint:
            plt.ylim((0,110))

        for metric in metric_df.columns:
            plt.plot(metric_df[metric], label=metric)
        plt.legend()
        plt.savefig(output_dir / f"{annotation_type}_metrics.png")

        metric_df.to_csv(str(output_dir / f"{annotation_type}_metrics.csv"))
        print(metric_df)

def create_validation_metrics(trial_dir, testing_json_path, testing_data_path, evaluation_metrics=True, n_graphed_images=-1, score_thresh=.9, keypoint=False):
    """Trial dir is a directory of .pth files"""
    register_coco_instances("real_test", {}, testing_json_path, testing_data_path)

    # if its an empty/nonexistent directory just calculate the imagenet values
    trial_dir.mkdir(exist_ok=True)
    weights_paths = [weights_path for weights_path in trial_dir.iterdir() if weights_path.suffix == ".pth"]
    if len(weights_paths) == 0:
        weights_paths = [None]
    
    if keypoint:
        output_dir = trial_dir / "keypoint_evaluations"
    else:
        output_dir = trial_dir / "evaluations"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(exist_ok=True)

    evaluations = {}
    n_epochs = 0
    for weights_path in weights_paths:
        # if weights_path.suffix != ".pth":
        #     continue

        # making directories
        curr_output_dir = output_dir / f"epoch_{n_epochs}"
        curr_image_output_dir = curr_output_dir / "example_annotations"
        curr_output_dir.mkdir(exist_ok=True, parents=True)
        curr_image_output_dir.mkdir(exist_ok=True, parents=True)

        cfg = setup_cfg(weights_path, score_thresh=score_thresh)
        predictor = DefaultPredictor(cfg)

        # getting example annotations
        graph_sample_images(predictor, testing_data_path, curr_image_output_dir, n_samples=n_graphed_images, gt_json_path=testing_json_path)

        # computing precision metrics
        if evaluation_metrics:
            evaluation = calc_single_trial_metrics(predictor, cfg, output_dir=curr_output_dir, score_thresh=score_thresh, keypoint=keypoint)
            evaluations[n_epochs] = dict(evaluation)
        n_epochs += 1

    process_evaluations(evaluations, output_dir, keypoint=keypoint)

def get_parser():
    parser = argparse.ArgumentParser(
        prog="Validation Metric Creator",
        description=""
    )

    parser.add_argument(
                        "-i",
                        "--input-path",
                        type=str,
                        help="Path to the directory of .pth checkpoints",
                        required=True
                        )

    parser.add_argument(
                        "-tjp",
                        "--testing-json-path",
                        type=str,
                        help="Path to the testing json for comparison/ground truth metrics",
                        required=True
                        )


    parser.add_argument(
                        "-tdp",
                        "--testing-data-path",
                        type=str,
                        help="Path to the testing data for comparison/ground truth metrics",
                        required=True
                        )

    parser.add_argument(
                        "-ni",
                        "--n-displayed-images",
                        type=int,
                        help="Number of images to visualize for annotation comparison.",
                        required=False,
                        default=-1
                        )

    parser.add_argument(
                        "-em",
                        "--evaluation-metrics",
                        type=bool,
                        help="Run the overall evaluation metrics",
                        required=False,
                        default=True
                        )

    parser.add_argument(
                        "-kp",
                        "--keypoint",
                        type=bool,
                        help="Run Keypoint evaluations instead.",
                        required=False,
                        default=False
                        )

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    in_path = Path(args.input_path)
    testing_json_path = Path(args.testing_json_path)
    testing_data_path = Path(args.testing_data_path)

    # in_path = r"/home/whuang/wheelpose/detectron2_scripts/output/models/B_1.5kRW_.00025lr_defaultHyperparams_30"
    # testing_json_path = r"/data/wheelpose/wheelpose_gt/B_test_1k.json"
    # testing_data_path = r"/data/wheelpose/wheelpose_gt/data"

    create_validation_metrics(in_path,
                                testing_json_path,
                                testing_data_path,
                                evaluation_metrics=args.evaluation_metrics,
                                n_graphed_images=args.n_displayed_images,
                                keypoint=args.keypoint)
