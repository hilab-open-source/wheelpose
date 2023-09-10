import argparse
from pathlib import Path
import json
import sys
import numpy as np
def sample_json(in_path, out_path, n_images, idx_range):
    with open(in_path, "r") as f:
        data = json.load(f)

    n_total_images = len(data["images"])

    if idx_range[1] == -1:
        idx_range[1] = n_total_images - 1

    idx_range_len = idx_range[1] - idx_range[0]
    if idx_range_len < n_images:
        sys.exit("Invalid number of images to sample. Too Large")

    sample_idx = np.round(np.linspace(idx_range[0], idx_range_len - 1, n_images)).astype(int)
    data["images"] = [data["images"][i] for i in sample_idx]

    selected_image_ids = [img["id"] for img in data["images"]] # already removed all the ones we done want
    
    new_annotations = []
    for ann in data["annotations"]:
        if ann["image_id"] in selected_image_ids:
            new_annotations.append(ann)
    
    data["annotations"] = new_annotations

    with open(out_path, "w") as f:
        json.dump(data, f, indent=4)
    
def get_parser():
    parser = argparse.ArgumentParser(
        prog="Sample COCO",
        description="Takes a subset of COCO images and creates a new json for testing. Subset is only by ordering and not the actual id"
    )

    parser.add_argument(
                        "-i", 
                        "--input-path", 
                        type=str, 
                        help="input path of JSON", 
                        required=True
                        )
    parser.add_argument(
                        "-o", 
                        "--output-path", 
                        type=str, 
                        help="output path of new JSON", 
                        required=True
                        )
    parser.add_argument(
                        "-n", 
                        "--n-images", 
                        type=int, 
                        help="Number of images to sample out", 
                        required=True
                        )
    parser.add_argument(
                        "-min", 
                        type=int, 
                        help="Minimum index of range to sample",
                        default=0
                        )
    parser.add_argument(
                        "-max", 
                        type=int, 
                        help="Maximum index of range to sample",
                        default=-1
                        )
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    sample_json(Path(args.input_path), Path(args.output_path), args.n_images, [args.min, args.max])
