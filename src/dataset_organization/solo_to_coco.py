import json
import cv2
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time
import os

import argparse

import STC_annotation_helper as ah

#keypoint annotation format - they split it up into 2 (with object detection), but the Keypoint annotations are the same except they just have a little extra info, so I'm just going to keep it as is
def get_json(input_path, output_path, save_depth, save_normal, save_seg, indent=False):
    """returns keypoint json jsons. Also optionally saves extra images
       Outlined in https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
       """

    _setup_output(output_path, save_depth, save_normal, save_seg)

    info, licenses, images, annotations, categories = _get_data(input_path, output_path, save_depth, save_normal, save_seg)

    keypoint_dict = {"info": info,
                    "images":images,
                    "annotations": annotations,
                    "categories": categories}

    if indent:
        indent_val = 4
    else:
        indent_val = 0
    keypoint_json = json.dumps(keypoint_dict, indent=indent_val)

    json_out_path = output_path / "labels.json"
    with open(json_out_path, "w+") as outfile:
        outfile.write(keypoint_json)

def _setup_output(output_path, save_depth, save_normal, save_seg):
    """creates output folder structure"""
    output_path.mkdir(exist_ok=True) # makes the output directory

    (output_path / "data").mkdir(exist_ok=True)

    if save_depth:
        (output_path / "segmentation_data").mkdir(exist_ok=True)

    if save_normal:
        (output_path / "normal_data").mkdir(exist_ok=True)

    if save_seg:
        (output_path / "segmentation_data").mkdir(exist_ok=True)

def _get_data(input_path, output_path, save_depth, save_normal, save_seg):
    """returns json data and also saves images to folder"""
    final_info = _get_info()
    final_licenses = [_get_licenses()]
    final_categories = [_get_categories()]
    final_images = []
    final_annotations = []


    annotation_id = 0
    # TODO: ENABLE COLELCTION OF MULTIPLE IMAGES PER SEQUENCE
    for image_id, folder in tqdm(enumerate(input_path.iterdir())):
        #load data
        if not folder.is_dir():
            continue
        frame_data_path = folder / "step0.frame_data.json"

        with open(frame_data_path, "r") as f:
            frame_data = json.load(f)

        if save_seg:
            segmentation_path = folder /  "step0.camera.instance segmentation.png"
            segmentation_img = cv2.imread(str(segmentation_path), cv2.IMREAD_UNCHANGED)
        else:
            segmentation_img = None # defaults to none to prevent segmentation reads

        #annotations
        try:
            annotation_data = frame_data['captures'][0]['annotations']
        except:
            print(f"{image_id} has no annotation capture data")
            continue

        metrics = frame_data['metrics']
        for d in metrics:
            type = d['@type']
            if type == "type.unity.com/unity.solo.OcclusionMetric":
                occulsion_data = d
                break

        try:
            visible_img_data = occulsion_data['values']
        except:
            continue

        if not visible_img_data:
            continue

        for v in visible_img_data:
            instance_id = v['instanceId']
            ann_instance = _get_annotations(annotation_id, instance_id, image_id, annotation_data, segmentation_img)
            final_annotations.append(ann_instance)
            annotation_id += 1

        #image

        img_dim = frame_data["captures"][0]["dimension"][::-1]
        img_path = folder / "step0.camera.png"
        img_instance = _get_images(image_id, str(image_id)+'.png', img_path, img_dim, frame_data)
        final_images.append(img_instance)

        _save_images(str(image_id), folder, output_path, save_depth, save_normal, save_seg)

    return final_info, final_licenses, final_images, final_annotations, final_categories

def _get_info():
    info_dict = {
        "description": "Solo Wheelchair Dataset",
        "version": "0.1",
        "year": 2023,
        "contributor": "",
        "date_created": datetime.today().strftime('%Y/%m/%d')
    }
    return info_dict

def _get_licenses():
    """
    "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    """

    license_instance_dict = {
        "url": None,
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License"
    }

    return license_instance_dict

def _get_categories():
    """
    "keypoints": [
            "nose","left_eye","right_eye","left_ear","right_ear",
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle"
        ],
        "skeleton": [
            [16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
            [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]
        ]
        """

    category_dict = {
        "supercategory": "person",
        "id": 1,
        "name": "person",
        "keypoints": ["nose","left_eye","right_eye","left_ear","right_ear",
                    "left_shoulder","right_shoulder","left_elbow","right_elbow",
                    "left_wrist","right_wrist","left_hip","right_hip",
                    "left_knee","right_knee","left_ankle","right_ankle"],
        "skeleton": [[16,14],[14,12],[17,15],[15,13],[12,13],[6,12],[7,13],[6,7],
                    [6,8],[7,9],[8,10],[9,11],[2,3],[1,2],[1,3],[2,4],[3,5],[4,6],[5,7]]
    }



    return category_dict

#for every image instance
def _get_images(image_id, file_name, file_path, dimensions, frame_data):
    """Params:
            image_id = image id (not to be confused with annotation id)
            file_name = file name (ex solo12312_camera.png)
            dimensions = tuple for (height, width) - technically im inputting (height, width, channels)
       """

    height = dimensions[0]
    width = dimensions[1]

    #img capture time
    ti_m = os.path.getmtime(file_path)
    m_ti = time.ctime(ti_m)
    t_obj = time.strptime(m_ti)
    T_stamp = time.strftime("%Y-%m-%d %H:%M:%S", t_obj)

    fov_val, focal_len = ah.get_image_fov_data(frame_data)

    images_dict = {
        "id": image_id,
        "license": 1,
        "file_name": file_name,
        "height": height,
        "width": width,
        "date_captured": str(T_stamp),
        "fov_value": fov_val,
        "camera_focal_length": focal_len
    }

    return images_dict

def _get_annotations(annotation_id, instance_id, image_id, annotation_data, segmentation_img):
    """
    Params:
        annotation_id = unique annotation id. This is incremented each time
        instance_id = SOLO id per image. this is used for logic in finding correct data per annotation. This value is NOT unique across the whole dataset.
        image_id = image id corresponding to what its from. Im just going to order each image numerically.
        annotation_data = loaded json for frame data
        segmentation_img = loaded segmentation img
    """

    num_keypoints, keypoints = ah.get_keypoints(instance_id, annotation_data)
    bbox_data = ah.get_bbox_data(instance_id, annotation_data)
    area = bbox_data[2] * bbox_data[3] # bbox area

    if segmentation_img is not None:
        # TODO: SEGMENTATION DATA CALCULATIONS HERE
        pass

    annotations_dict = {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": 1,
        "iscrowd": 0,
        "area": area,
        "bbox": bbox_data,
        "keypoints": keypoints,
        "num_keypoints": num_keypoints
    }
    #        "segmentation": segmentation_data,
    return annotations_dict

def _copy_file(input_path, output_dir, file_name):
    """helper function to copy file from input to output directory with specified file name maintaining extension"""
    output_path = output_dir / f"{file_name}{input_path.suffix}"

    shutil.copy(input_path, output_path)

def _save_images(file_name, input_path, output_path, save_depth, save_normal, save_seg):
    # copy rgb no matter what
    rgb_path = input_path / "step0.camera.png"
    _copy_file(rgb_path, output_path / "data", file_name)

    if save_depth:
        depth_path = input_path / "step0.camera.Depth.exr"
        _copy_file(depth_path, output_path / "depth_data", file_name)

    if save_normal:
        normal_path = input_path / "step0.camera.Normal.exr"
        _copy_file(normal_path, output_path / "normal_data", file_name)

    if save_seg:
        segmentation_path = input_path / "step0.camera.instance segmentation.png"
        _copy_file(segmentation_path, output_path / "segmentation_data", file_name)

def get_parser():
    parser = argparse.ArgumentParser(
        prog="Solo To COCO Data conversion",
        description="Converts SOLO generated data from Unity Perception to COCO"
    )

    parser.add_argument(
                        "-i",
                        "--input-path",
                        type=str,
                        help="input directory of SOLO",
                        required=True
                        )
    parser.add_argument(
                        "-o",
                        "--output-path",
                        type=str,
                        help="output of COCO",
                        required=True
                        )
    parser.add_argument(
                        "-sn",
                        "--save-normal",
                        type=bool,
                        help="Saves normal frames",
                        default=False
                        )
    parser.add_argument(
                        "-ss",
                        "--save-seg",
                        type=bool,
                        help="Saves segmentation frames",
                        default=False
                        )
    parser.add_argument(
                        "-sd",
                        "--save-depth",
                        type=bool,
                        help="Saves depth frames",
                        default=False
                        )

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    args = vars(args)
    args["input_path"] = Path(args["input_path"])
    args["output_path"] = Path(args["output_path"])
    # if args.in_path:
    #     IN_FOLDER = os.path.abspath(args.in_path)
    # if args.out_path:
    #     OUT_FOLDER = os.path.abspath(args.out_path)

    get_json(**args, indent=True)
