"""Takes the first coco annotation file and then merges it with the second, 
replacing all annotations per image in the first with the ones in the second. 
Make sure naming is aligned (doesnt need to be same order)"""

from pathlib import Path
import sys
import argparse
import json
from tqdm import tqdm

ALT_KP_NAMES = {
    "wristl": "left_wrist",
    "wristr": "right_wrist",
    "elbowl": "left_elbow",
    "elbowr": "right_elbow",
    "shoulderl":"left_shoulder",
    "shoulderr": "right_shoulder",
    "earl": "left_ear",
    "earr": "right_ear",
    "eyel": "left_eye",
    "eyer": "right_eye",
    "hipl": "left_hip",
    "hipr": "right_hip",
    "kneel": "left_knee",
    "kneer": "right_knee",
    "anklel": "left_ankle",
    "ankler": "right_ankle"
}

def get_kp_names(ann, selected_cat="Human"):
    if "keypoints" not in ann:
        for cat in ann["categories"]:
            if cat["name"] == selected_cat:
                return cat["keypoints"]
        sys.exit("Selected category not in annotation file")
    else:
        return ann["keypoints"]

def get_ann_img(ann, img_id):
    # gets the indices of all annotations on a specified image
    anns = []
    for i, a in enumerate(ann["annotations"]):
        if a["image_id"] == img_id:
            anns.append(a)
    return anns

def get_img(ann, img_id):
    # gets the image name from image idx
    for img in ann["images"]:
        if img["id"] == img_id:
            return img
    return None

def get_img_id(ann, img_file_name):
    for img in ann["images"]:
        if img["file_name"] == img_file_name:
            return img
    return None

def reorder_keypoints(kp_arr, base_order, new_order):
    for i in range(len(new_order)):
        if new_order[i] in ALT_KP_NAMES:
            new_order[i] = ALT_KP_NAMES[new_order[i]]

    new_kp_ordering_dict = {new_order[i]:i for i in range(len(new_order))}
    new_kp_arr = []
    for kp in base_order:
        new_kp_idx = new_kp_ordering_dict[kp]
        new_kp_arr.extend(kp_arr[new_kp_idx*3:(new_kp_idx+1)*3])

    return new_kp_arr

def main(args):
    base_path = Path(args.base)
    joined_path = Path(args.joined)
    out_path = Path(args.output)

    with open(base_path, "r") as f:
        base_ann = json.load(f)
    
    with open(joined_path, "r") as f:
        joined_ann = json.load(f)

    joined_kp_names = get_kp_names(joined_ann)
    base_kp_names = get_kp_names(base_ann)

    # getting the highest id so they're all unique
    ids = []
    for a in base_ann["annotations"]:
        ids.append(a["id"])

    max_base_id = max(ids)

    curr_id = max_base_id + 1 

    img_id_to_remove = []
    for i in range(len(joined_ann["annotations"])):
        curr_ann = joined_ann["annotations"][i]

        img = get_img(joined_ann, curr_ann["image_id"])
        img_name = img["file_name"]

        base_img = get_img_id(base_ann, img_name)
        base_img_id = base_img["id"]

        if base_img_id is None:
            print(f"Missing image: {img_name}. Continuing")

        img_id_to_remove.append(base_img_id)

        reordered_kp = reorder_keypoints(curr_ann["keypoints"], base_kp_names, joined_kp_names)

        curr_ann["id"] = curr_id
        curr_ann["image_id"] = base_img_id
        curr_ann["category_id"] = 0
        curr_ann["keypoints"] = reordered_kp
        curr_ann["num_keypoints"] = int(len(reordered_kp) / 3)
        
        base_ann["annotations"].append(curr_ann)
        curr_id += 1

    for a in base_ann["annotations"][:max_base_id]:
        if a["image_id"] in img_id_to_remove:
            print(a["id"])
            base_ann["annotations"].remove(a)
    
    with open(out_path, "w") as f:
        json.dump(base_ann, f, indent=4)

def get_parser():
    parser = argparse.ArgumentParser(
        prog="Merge COCO ",
    )

    parser.add_argument(
                        "-b", 
                        "--base", 
                        type=str, 
                        help="base coco file to take naming from", 
                        required=True
                        )
    
    parser.add_argument(
                        "-j", 
                        "--joined", 
                        type=str, 
                        help="joined coco file", 
                        required=True
                        )
    
    parser.add_argument(
                        "-o", 
                        "--output", 
                        type=str, 
                        help="output path", 
                        required=True
                        )

    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)