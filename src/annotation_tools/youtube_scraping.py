import numpy as np
import cv2
from pytube import YouTube
import os
from tqdm import tqdm

def scrape_one_video(video_dir, save_per_sec=8, time_tuple=None, output_dir=None):
    # TODO: allow you to take in a list of frames instead

    """gets frames every save_per_sec at start and end times. Takes in a local video file"""
    filename, _ = os.path.splitext(video_dir)
    filename += f"-fps{save_per_sec}"
    # make a folder by the name of the video file
    if not os.path.isdir(filename):
        os.mkdir(filename)
    cap = cv2.VideoCapture(video_dir)

    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_per_sec = min(fps, save_per_sec)

    # get the list of duration spots to save
    if time_tuple:
        assert len(time_tuple)==2
        start_frame, end_frame = _get_frame_ends(fps, time_tuple)
    else:
        start_frame = 0
        end_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT) + 1 #total frames+1

    frame_steps = np.arange(start_frame, end_frame, save_per_sec)

    for frame_num in frame_steps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num-1) #-1 because videocapture is dumb
        is_read, frame = cap.read()
        if not is_read:
            break #frame doesnt exist

        cv2.imwrite(os.path.join(filename, f"frame{int(frame_num)}.jpg"), frame)


def from_youtube_link(link, raw_folder_dir, save_per_sec=8, time_tuple=None, output_dir=None):
    video_dir, title = download_raw_youtube(link, raw_folder_dir=raw_folder_dir)
    scrape_one_video(video_dir, save_per_sec=save_per_sec, time_tuple=time_tuple, output_dir=output_dir)

def download_raw_youtube(link, raw_folder_dir, resolution=720):
    """downloads youtube video at specified resolution from link"""
    #IMPORTANT - need to replace pytube cipher.py line 30 with: var_regex = re.compile(r"^\$*\w+\W")

    yt = YouTube(link)
    res = f"{resolution}p"
    try:
        ys = yt.streams.get_by_resolution(res) #get stream obj at resolution
        if not ys:
            print(f"could not find {link} with specified resolution {res}")
            ys = yt.streams.get_highest_resolution()
    except Exception as e: # age restricted or video not found errors
        print(e, flush=True)
        return None, None

    video_path = ys.download(raw_folder_dir, filename_prefix="raw-", skip_existing=True)
    #download video, can maybe implement code to delete after frame scrape later
    print(f"Successfully Downloaded Youtube Link: {link}")
    return video_path, yt.title

def from_folder(folder_dir, sps_list=None, tt_list=None, output_dir=None):
    """takes in directory of video files"""
    for i, video_dir in tqdm(enumerate(os.listdir(folder_dir))):
        if sps_list:
            try:
               save_per_sec = sps_list[i]
            except:
               save_per_sec=None
        if tt_list:
            try:
                time_tuple = tt_list[i]
            except:
                time_tuple = None

        scrape_one_video(video_dir, save_per_sec=save_per_sec, time_tuple=time_tuple, output_dir=output_dir)

def _get_frame_ends(fps, time_tuple):
    """gets start and end frame given time elapsed in seconds from (starttime, endtime)"""
    start_time = time_tuple[0] #in seconds
    end_time = time_tuple[1]

    assert start_frame >=0
    assert start_frame < end_frame

    start_frame = start_time*fps
    end_frame = end_time*fps

    return start_frame, end_frame


if __name__ == "__main__":
    save_per_sec = 4 #frames collected per second
    import sys

    video_file = sys.argv[1]
    from_youtube_link(video_file, raw_folder_dir=sys.argv[2])
