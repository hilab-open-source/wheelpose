import sys
import argparse
from pathlib import Path
from multiprocessing import Queue
import cv2
import os
import shutil
import signal

from process import Process
import youtube_scraping as ys

DEFAULT_ROW = {
            "file_name": "",
            "video_id": -1,  # hash the url for an id
            "url": "",
            "title": "",
            "frame_number": -1,
            "multipose": False,
            "img": None, # IMAGE ARRAY FOR SAVING
            "n_frames": -1, # METADATA
            }

def process_in_file(in_file):
    in_file = Path(in_file)
    if not in_file.exists():
        print(f"In file {in_file} does not exist")
        sys.exit()

    with open(in_file, "r") as f:
        urls = f.readlines()

    return urls

def process_out_dir(out_dir):
    # creates all the necessary save paths
    out_dir = Path(out_dir)
    video_dir = out_dir / "videos"
    frame_dir = out_dir / "frames"

    video_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, video_dir, frame_dir

def get_frame_numbers(n_frames, fps, sample_step, max_samples):
    sample_step = sample_step * (fps / 60) # sample more closely if theres less frames

    if sample_step * max_samples < n_frames:
        # just sample across the video equally if there are more than enough frames
        step = int(n_frames / max_samples)
        return [i * step for i in range(max_samples)]
    else: # this is literally np.linspace im just too lazy to import numpy
        frames = []
        curr_frame = 0
        while curr_frame < n_frames:
            frames.append(curr_frame)
            curr_frame += sample_step
        return frames

def video_frames_producer(queue, urls, video_dir, local_videos, res, sample_step, max_samples):
    print("Producer: Running", flush=True)
    for url in urls:
        if not local_videos:
            p, title = ys.download_raw_youtube(url, str(video_dir), resolution=res)
            if p is None: # skips url
                print(f"Could not access {url}. Continuing", flush=True)
                continue
        else: # using local videos
            p = str(video_dir / str(url.name))
            print(p, flush = True)
            title = str(url.stem)
            shutil.copyfile(str(url), str(p)) # copies videos into video folder

        cap = cv2.VideoCapture(p)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        sampled_frames = get_frame_numbers(n_frames, fps, sample_step, max_samples)

        for frame_num in sampled_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1)
            is_read, frame = cap.read()
            if not is_read:
                break # frame doesnt exist
            
            # out_data = {
            #             "url": url,
            #             "video_title": title,
            #             "id": str(abs(hash(url)))[:10],  # hash the url for an id
            #             "frame_number": int(frame_num),
            #             "img": frame,
            #             "n_frames": int(n_frames),
            #             }

            out_data = DEFAULT_ROW.copy()
            out_data["url"] = url
            out_data["title"] = title
            out_data["video_id"] = str(abs(hash(url)))[:10]
            out_data["frame_number"] = int(frame_num)
            out_data["img"] = frame
            out_data["n_frames"] = int(n_frames)

            queue.put(out_data)
        print(f"Finished Downloading {title}", flush=True)
    
    queue.put(None)
    print("Producer: Done", flush=True)

def save_consumer(queue, out_dir, frame_dir):
    print("Save Process: Running", flush=True)
    col_names = list(DEFAULT_ROW.keys())[:-2] # -2 to cut out metadata stuff
    record_file = out_dir / "frames.txt"
    if not record_file.exists():
        col_header = ",".join([col for col in col_names]) + ",\n"
        with open(record_file, "w") as f:
            f.write(col_header)

    while True:
        item = queue.get()
        if item is None:
            break
        
        m = "m" if item["multipose"] else "s" # multipose marker
        file_name = f"{item['video_id']}_{item['frame_number']}_{m}.png"
        item["file_name"] = file_name
        # save the image
        cv2.imwrite(str(frame_dir / file_name), item["img"])

        # make a string of the data
        record = [str(item[col]).rstrip() for col in col_names]
        record = ",".join([val for val in record]) + ",\n" # adds the new line delimiter
        # record the metadata to a file
        with open(record_file, "a", encoding="utf-8") as f:
            f.write(record)

        print(f"Saved {file_name}", flush=True)
    
    print("Save Process: Exited", flush=True)

def display_frames_consumer(queue, out_dir, frame_dir):
    print("Consumer: Running", flush=True)

    save_queue = Queue(maxsize=50)

    # multiprocess the saving
    save_process = Process(target=save_consumer,
                            args=(save_queue, out_dir, frame_dir))
    save_process.daemon = True
    save_process.start()

    prev_video_id = -1
    multipose = False
    while True: # go through the queue till its gone
        item = queue.get()
        # print(item, flush=True)
        if item is None:
            break
        
        display_text = f"{item['title']}: Frame {item['frame_number']} / {item['n_frames']}"
        print(display_text, flush=True)

        # multipose toggling stuff
        if (prev_video_id != item["video_id"]):
            print("New Video. Setting Multipose Toggle to False.", flush=True)
            multipose = False

        cv2.imshow("Frame Displayer",item["img"])
        save = False
        while True:
            k = cv2.waitKey(0)
            if k == ord("d"):
                break
            elif k == ord("s"):
                save = False
                print("Marked to skip", flush=True)
            elif k == ord("w"):
                save = True
                print("Marked to save", flush=True)
            elif k == ord("q"):
                multipose = True
                print("Multipose Toggled to True. This state will maintain for the rest of the video unless specified", flush=True)
            elif k == ord("e"):
                multipose = False
                print("Multipose Toggled to False. This state will maintain for the rest of the video unless specified", flush=True)

        item["multipose"] = multipose
        prev_video_id = item["video_id"]

        if save: # save if marked
            save_queue.put(item)

    save_queue.put(None) # ends save queue

    save_process.join()
    if save_process.exception:
        raise save_process.exception
    print("All Images Analyzed", flush=True)

def main(args):
    if not args.local_videos:
        urls = process_in_file(args.input)
    else:
        urls = [f for f in Path(args.input).iterdir()]

    out_dir, video_dir, frame_dir = process_out_dir(args.output)
    frame_queue = Queue(maxsize=2*args.max_frames)

    consumer_process = Process(target=display_frames_consumer, 
                            args=(frame_queue, 
                                    out_dir, 
                                    frame_dir,
                                )
                            )

    producer_process = Process(target=video_frames_producer, 
                            args=(  frame_queue, 
                                    urls, 
                                    video_dir, 
                                    args.local_videos,
                                    args.resolution, 
                                    args.sample_rate, 
                                    args.max_frames,
                                    )
                            )

    # handling keyboard interrupt
    def signal_handler(sig, frame):
        print("Forced Closed Program \n\n\n", flush=True)
        # producer_process.terminate()
        # consumer_process.terminate()
        # sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    producer_process.start()
    consumer_process.start()


    # close processes when done
    producer_process.join()
    if producer_process.exception:
        consumer_process.kill()
        raise producer_process.exception
    consumer_process.join()
    if consumer_process.exception:
        producer_process.kill()
        raise consumer_process.exception

    if args.delete_videos:
        for p in video_dir.iterdir():
            os.remove(p)

def get_parser():
    parser = argparse.ArgumentParser(
        prog="Youtube Image Downloader",
        description="Samples a selection of frames from youtube videos listed in input text file. Saves out a set of frames"
    )

    parser.add_argument(
                        "-i", 
                        "--input", 
                        type=str, 
                        help="input file of urls", 
                        required=True
                        )
    parser.add_argument(
                        "-o", 
                        "--output", 
                        type=str, 
                        help="output directory", 
                        required=True
                        )
    parser.add_argument(
                        "-s", 
                        "--sample-rate", 
                        type=int, 
                        help="Minimum number of frames to delay between given 60 fps. Scaled accordingly depending on video fps",
                        default=60
                        )
    parser.add_argument(
                        "-m", 
                        "--max-frames", 
                        type=int, 
                        help="Max number of frames per video to select",
                        default=500
                        )
    parser.add_argument(
                        "-r", 
                        "--resolution", 
                        type=int, 
                        help="Preferred resolution of videos",
                        default=720
                        )
    
    parser.add_argument(
                        "-d", 
                        "--delete-videos", 
                        type=bool, 
                        help="Delete videos after downloading them, keeping only the frames",
                        default=False
                        )
    
    parser.add_argument(
                        "-l", 
                        "--local-videos", 
                        type=bool, 
                        help="Input is treated as a folder of raw videos instead of a text file of links",
                        default=False
                        )
    return parser

if __name__ == "__main__":
    """Returns a save directory with a file of metadata including.
    """
    parser = get_parser()

    args = parser.parse_args()
    main(args)