import torch
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import argparse
import os

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

from detectron2.engine import DefaultTrainer
from detectron2.engine import launch

from custom_classes import CustomTrainer, ValLossScheduler


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

# Pairs of keypoints that should be exchanged under horizontal flipping
COCO_PERSON_KEYPOINT_FLIP_MAP = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)

def main(args):
    print('main')
    #register dataset
    datasets=['train']
    register_coco_instances("train", {}, args.json_path, args.image_path)
    if args.anneal:
        datasets.append('val')
        register_coco_instances("val", {}, args.val_json_path, args.val_image_path)

    train(args, datasets)

def train(args, datasets):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")) #50, 101
    cfg.DATASETS.TRAIN = ("train",)
    if args.anneal:
        cfg.DATASETS.VAL = ("val",)
        cfg.TEST.EVAL_PERIOD = 1

    cfg.DATASETS.TEST = ()

    cfg.DATALOADER.NUM_WORKERS = args.num_workers

    cfg.SEED = args.seed

    if args.weights:
        cfg.MODEL.WEIGHTS = os.path.abspath(args.weights)
    else:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml") #50

    if args.from_scratch:
        cfg.MODEL.WEIGHTS = ""

    cfg.SOLVER.IMS_PER_BATCH = args.batch_size  # This is the real "batch size"
    cfg.SOLVER.BASE_LR = args.learning_rate  # pick a good LR

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = args.loss_batch_size
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = args.num_classes

    cfg.SOLVER.STEPS = [] #dont use default detectron lr annealing
    cfg.SOLVER.WARMUP_ITERS = 0
    cfg.SOLVER.WARMUP_ITERS = 1000 #DEELTTE
    if args.iterations:
        cfg.SOLVER.MAX_ITER = args.iterations
        cfg.SOLVER.CHECKPOINT_PERIOD = args.iterations//4
    else:
        #calculating iterations
        one_epoch = len(DatasetCatalog.get("train")) // args.batch_size #num training/batch size
        max_iter = one_epoch * args.epochs #num epochs
        max_iter = int(max_iter)

        cfg.SOLVER.MAX_ITER = max_iter
        print(max_iter)
        cfg.SOLVER.CHECKPOINT_PERIOD = int(one_epoch)
        print(f"SOLVER CHECKPOINT PERIOD: {int(one_epoch)}")

        if args.step_anneal:
            print(args.epochs)
            #for i in range(1, (args.epochs//10) - 3):
            #    cfg.SOLVER.STEPS.append(one_epoch*i*10)
            #print(cfg.SOLVER.STEPS)
            cfg.SOLVER.STEPS.append(one_epoch*20)
            print(cfg.SOLVER.STEPS)

    for d_name in datasets:
        MetadataCatalog.get(d_name).keypoint_names = COCO_PERSON_KEYPOINT_NAMES
        MetadataCatalog.get(d_name).keypoint_flip_map = COCO_PERSON_KEYPOINT_FLIP_MAP

    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.anneal:
        trainer = CustomTrainer(cfg)
    else:
        trainer = DefaultTrainer(cfg)

    trainer.resume_or_load(resume=False)
    trainer.train()


if __name__ == "__main__":
    print('start')
    """Returns a save directory with a file of metadata including.
    """

    parser = argparse.ArgumentParser(
        prog="basic default trainer",
        description=""
    )
    parser.add_argument(
                        "-jp",
                        "--json-path",
                        type=str,
                        help="path to model json coco file",
                        required=True
                        )
    parser.add_argument(
                        "-ip",
                        "--image-path",
                        type=str,
                        help="path to coco model images. Must end in \{model_name}.pth",
                        required=True
                        )
    parser.add_argument(
                        "-o",
                        "--output-dir",
                        type=str,
                        help="output where final weights will be stored",
                        required=True
                        )
    parser.add_argument(
                        "-nw",
                        "--num-workers",
                        type=int,
                        help="number of parallel data loading workers",
                        default=2
                        )
    parser.add_argument(
                        "-bs",
                        "--batch-size",
                        type=int,
                        help="training batch size. Keep as a power of 2",
                        default=4
                        )
    parser.add_argument(
                        "-lr",
                        "--learning-rate",
                        type=float,
                        help="Learning rate",
                        default=0.00025
                        )
    parser.add_argument(
                        "-e",
                        "--epochs",
                        type=int,
                        help="number of training epochs",
                        default=1
                        )
    parser.add_argument(
                        "-i",
                        "--iterations",
                        type=int,
                        help="specific training iterations. This will override epoch number",
                        default=None
                        )
    parser.add_argument(
                        "-lbs",
                        "--loss-batch-size",
                        type=int,
                        help="used to sample a subset of proposals coming out of RPN to calculate cls and reg loss during training",
                        default=512
                        )
    parser.add_argument(
                        "-nc",
                        "--num-classes",
                        type=int,
                        help="number of classes in model",
                        default=1
                        )
    parser.add_argument(
                        "-w",
                        "--weights",
                        type=str,
                        help="Path to the starting model weights. If none, we use the COCO r50 model weights.",
                        default=None
                        )
    parser.add_argument(
                        "-ann",
                        "--anneal",
                        type=bool,
                        help="Flag for LR annealing on Validation Plateau. Default is false.",
                        default=False
                        )
    parser.add_argument(
                        "-val-jp",
                        "--val-json-path",
                        type=str,
                        help="path to coco json for validation set. Used for LR annealing",
                        default=False
                        )
    parser.add_argument(
                        "-val-ip",
                        "--val-image-path",
                        type=str,
                        help="path to image folder for validation set. Used for LR annealing",
                        default=False
                        )
    parser.add_argument(
                        "-lr-f",
                        "--lr-factor",
                        type=float,
                        help="LR Annealing param. Factor by which the learning rate will be reduced",
                        default=0.1
                        )
    parser.add_argument(
                        "-lr-p",
                        "--lr-patience",
                        type=int,
                        help="LR Annealing param. The number of epochs with no validation improvement after which learning rate will be reduced by lr-factor",
                        default=2
                        )
    parser.add_argument(
                        "-lr-cd",
                        "--lr-cooldown",
                        type=int,
                        help="Lr Annealing param. Number of epochs to wait before resuming normal operation after lr has been reduced",
                        default=0
                        )
    parser.add_argument(
                        "-lr-min",
                        "--lr-minimum",
                        type=float,
                        help="LR Annealing param. The lower bound on the learning rate.",
                        default=0
                        )
    parser.add_argument(
                        "-lr-th",
                        "--lr-threshold",
                        type=float,
                        help="LR Annealing param. The minimum value of which the loss should change to count as an improvment. ex. Threshold=0.01 and Val loss goes from 0.4 -> 0.4001 this is below the threshold and not an improvment",
                        default=0.01
                        )
    parser.add_argument(
                        "-s-ann",
                        "--step-anneal",
                        type=bool,
                        help="If you want to use detectron2 LR annealing every 10 epochs",
                        default=False
                        )
    parser.add_argument(
                        "-scratch",
                        "--from-scratch",
                        type=bool,
                        help="Train from scratch",
                        default=False
                        )
    parser.add_argument(
                        "-seed",
                        "--seed",
                        type=int,
                        help="Set random seed. -1 by default (no seed)",
                        default=-1
                        )
    parser.add_argument(
                        "-ngpu",
                        "--n-gpus",
                        type=int,
                        help="Number of GPUS to train on",
                        default=4
                        )


    args = parser.parse_args()

    if ('anneal' in vars(args) and
        'val_image_path' not in vars(args) or
        'val_json_path' not in vars(args)):
        parser.error('LR Annealing (-anneal) requires a valid validation set (-val-ip and -val-jp')

    launch(
        main,
        num_gpus_per_machine=args.n_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=[args] #must be iterator
    )
