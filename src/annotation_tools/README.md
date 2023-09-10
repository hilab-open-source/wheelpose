# IMPORTANT
In current version of PyTube, there is a bug in the regex. Replace line 30 of ```cipher.py``` in the PyTube library with the following:
```
var_regex = re.compile(r"^\$*\w+\W")
```

# Manual Classification
Takes a file of a list of youtube urls and downloads them. Allows a user to sample across individual frames and save out the frame with information on the YouTube video's metadata.

## Usage
Run
```
python manual_classification.py -i input_file -o output_directory
```

Flags:
```
-i, --input - input file of urls
-o, --output - output directory
-s --sample-rate - minimum number of frames between each sample
-m --max-frames - max number of frames to sample from each video
-r --resolution - preferred resolution to save videos
-d --delete-videos - clear videos after downloading and processing them
```
Press "w" to mark a frame to save, "s" to mark as skip, "d" to go to next frame

The input file is a list of youtube urls delimited by new lines.
The output directory saves videos into the following
- Video Folder: temporary folder to store saved videso
- Frames Folder: Folder of saved Frames
- frames.txt: File of meta data. Organized as file_name, video_id, url, title, frame_number
