import argparse 
from pathlib import Path
from multiprocessing import Queue
import cv2
import sys
import json
import numpy as np

from process import Process


def get_img_dict(labels):
    # easier indexing for a specific image
    image_dict = {}
    for img in labels["images"]:
        image_dict[img["id"]] = img["file_name"]

    return image_dict

def image_producer(queue, labels, img_path):
    img_dict = get_img_dict(labels)
    annotations = labels["annotations"]

    for i, annotation in enumerate(annotations):
        img_file_name = img_dict[annotation["image_id"]]
        img = cv2.imread(str(img_path / img_file_name))

        # adding keypoints to check which id is which
        kps = np.array(annotation["keypoints"]).reshape(-1, 3)
        for kp in kps:
            if kp[2] != 2:
                continue
            kp = (int(kp[0]), int(kp[1]))
            img = cv2.circle(img, kp, 5, (255,255,255), -1)
        
        org = (5, img.shape[0]-5)
        img = cv2.putText(
                        img, 
                        f"Instance ID: {annotation['id']}", 
                        org, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (255,255,255), 
                        2, 
                        cv2.LINE_AA
                        )

        annotation["img"] = img

        annotation["index"] = i
        annotation["n_annotations"] = len(annotations)

        queue.put(annotation)
    print(f"Finished Loading All Images", flush=True)

    queue.put(None)
    print("Producer: Done", flush=True)

def calc_bbox(mouse_pos) :
    # list of tuples
    # top left corner is 0,0

    mouse_pos = np.array(mouse_pos)

    top_left = np.min(mouse_pos, axis=0)
    width, height = np.abs(mouse_pos[1] - mouse_pos[0])

    return [int(top_left[0]), int(top_left[1]), int(width), int(height)]

def display_bbox(img, bbox, color=(0,0,255)):
    start_point = int(bbox[0]), int(bbox[1])
    end_point = int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])
    img = cv2.rectangle(img, start_point, end_point, color, 3)

    # return img
    cv2.imshow("Image Displayer", img)

mouse_down = False
mouse_pos_list = []
def mouse_callback(event, x, y, flags, annotation):
    global mouse_pos_list # limited to size of 2
    global mouse_down
    img = annotation["img"]

    # ensure xy doesnt leave image dimensions
    x = np.clip(x, 0, img.shape[1])
    y = np.clip(y, 0, img.shape[0])

    if event == cv2.EVENT_LBUTTONDOWN: # click down management
        if len(mouse_pos_list) >= 2:
            cv2.imshow("Image Displayer", img)
            mouse_pos_list.clear()
        
        mouse_down = True
        mouse_pos_list.append((x,y))

    elif (event == cv2.EVENT_LBUTTONUP) and mouse_down: # final box
        mouse_down = False
        mouse_pos_list.append((x,y))
        img_boxed = img.copy()
        cv2.rectangle(img_boxed, mouse_pos_list[0], mouse_pos_list[1], (50,205,50), 2)
        cv2.imshow("Image Displayer", img_boxed)

    elif (event == cv2.EVENT_MOUSEMOVE) and mouse_down: # drag box
        img_hover = img.copy()
        cv2.rectangle(img_hover, mouse_pos_list[0], (x,y), (50,205,50), 2)
        cv2.imshow("Image Displayer", img_hover)

def image_viewer(queue, args):
    cv2.namedWindow("Image Displayer")

    bboxes = {}

    while True:
        annotation = queue.get()
        if annotation is None:
            break
        cv2.setMouseCallback("Image Displayer", mouse_callback, annotation)
        
        display_text = f"Instance ID: {annotation['id']} Image ID: {annotation['image_id']} Annotation: {annotation['index']+1}/{annotation['n_annotations']}"
        print(display_text, flush=True)
        img = annotation["img"]

        if args.display_orig:
            display_bbox(img, annotation["bbox"], (50,255,50))
            print(f"Original Bounding Box: {annotation['bbox']}", flush=True)

        mouse_pos_list.clear()

        cv2.imshow("Image Displayer", img)
        while True:
            k = cv2.waitKey(20) & 0xFF

            if k == ord(" "):
                if len(mouse_pos_list) != 2:
                    if args.skip:
                        print("NO NEW ANNOTATION. SKIPPING", flush=True)
                        break # can enable to allow for skipping
                    else:
                        print("PLEASE SELECT TWO CORNER POINTS FOR BBOX", flush=True)
                else:
                    # print(f"CORNER POSITIONS: {mouse_pos_list}")
                    bbox = calc_bbox(mouse_pos_list)
                    print(f"Instance ID {annotation['id']} Bounding Box: {bbox}")
                    bboxes[annotation["id"]] = bbox
                    
                    # testing to make sure the bboxes look right
                    # img_boxed = img.copy()
                    # display_bbox(img_boxed, bbox)
                    # display_bbox(img_boxed, annotation["bbox"], color=(0,255,0))
                    break
    
    return bboxes

def save_bboxes(save_path, labels, bboxes):

    print(f"Saving the following Bounding Boxes:\n {bboxes}")
    for annotation in labels["annotations"]:
        ann_id = annotation["id"]
        if ann_id not in bboxes:
            continue

        annotation["bbox"] = bboxes[ann_id]

    with open(save_path, "w") as f:
        json.dump(labels, f, indent=4)

def main(args):
    in_path = Path(args.input)
    labels_path = in_path / "labels.json"
    img_path = in_path / "data"
    
    with open(labels_path, "r") as f:
        labels = json.load(f)

    img_queue = Queue(maxsize=10)
    producer_process = Process(target=image_producer,
                            args=(img_queue,
                                labels,
                                img_path
                                )
                            )
    producer_process.daemon = True
    producer_process.start()
    
    bboxes = image_viewer(img_queue, args)

    if args.output == "":
        args.output = "labels.json"
    save_path = in_path / args.output

    if args.save:
        save_bboxes(save_path, labels, bboxes)

    producer_process.join()
    if producer_process.exception:
        print(str(producer_process.exception[1]))
        sys.exit()

def get_parser():
    parser = argparse.ArgumentParser(
        prog="COCO BBOX Editor",
        description="Editing tool to fix COCO bounding boxes"
    )

    parser.add_argument(
                        "-i", 
                        "--input", 
                        type=str, 
                        help="input directory of coco", 
                        required=True
                        )
    
    parser.add_argument(
                        "-o", 
                        "--output", 
                        type=str, 
                        help="output json file name. Default overwrites original json", 
                        default=""
                        )
    
    parser.add_argument(
                        "-d", 
                        "--display-orig", 
                        type=bool, 
                        help="Displays the original bounding box", 
                        default=False
                        )
    
    parser.add_argument(
                        "-s", 
                        "--skip", 
                        type=bool, 
                        help="Allows user to skip without annotating the image", 
                        default=False
                        )
    
    parser.add_argument(
                        "-sa", 
                        "--save", 
                        type=bool, 
                        help="Saves the bboxes to the output file", 
                        default=True
                        )

    return parser

if __name__ == "__main__":
    """HAS NO REALTIME SAVING SO GOOD LUCK"""
    parser = get_parser()
    args = parser.parse_args()
    main(args)