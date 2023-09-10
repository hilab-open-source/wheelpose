# WheelPose: Data Synthesis Techniques to Improve Pose Tracking Performance on Wheelchair Users

### [[Project Page]](<LINK>) [[Paper]](<LINK>)

![Teaser Image](./docs/example_synthetic_data.png)

![Static Badge](https://img.shields.io/badge/license-MIT-green)
&nbsp;
![unity_version](https://img.shields.io/badge/unity-2021.3.19f1-green)
## Summary
- We introduce WheelPose, a synthetic data generator with parametrized domain randomization for wheelchair users extended from Unity's [PeopleSansPeople](https://unity-technologies.github.io/PeopleSansPeople/).
- WheelPose consists of a full pipeline to generate human-centric synthetic data including the following features.
  - Conversion and pose modification of existing motion capture and motion sequences in XYZ joint format.
  - Conversion of motion sequences expressed as joint rotations into fully functional FBX models and Unity *AnimationClips*.
  - Simulation-ready and configurable Unity environment with Unity [SyntheticHumans](https://github.com/Unity-Technologies/com.unity.cv.synthetichumans) integration along with parametrized backgrounds, lighting, camera systems, and more with support for custom parameters.
  - Full support for all default Unity [Perception 1.0.0](https://github.com/Unity-Technologies/com.unity.perception) labelers including 2D and 3D bounding boxes, 2D and 3D keypoint positions, instance and semantic segmentation, depth, surface normal, occlusion, instance count, and more with support for custom labelers.
  - Preset naïve ranges for the domain randomization.
- We provide a set of tools for data analysis and cleaning.
- We provide a fast annotator tool for the collection of individual frames from YouTube videos.
- We provide a training methodology and framework for [Detectron2](https://github.com/facebookresearch/detectron2) and found noticeable improvements in bounding box and keypoint performance when using WheelPose and testing on real wheelchair users compared to ImageNet.

We provide all relevant code within this repo.
**If any code has flags, a user can run `python FILE.py -h` to see more information on potential command line options.**

## Installation

### Cloning via Command Line

Prior to cloning via command line, ensure that you have [Git LFS](https://git-lfs.com/) installed, otherwise large files will not download correctly.

You can clone the repository with:
```
git clone REPO_URL_HERE
```

### Python Environment
All Python code was developed in **Python 3.10.8**. Dependencies can be installed in the following through pip.

```
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # CUDA 11.8. Install Torch based on your local machine
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```


If you cannot successfully create the environment, here is a list of required libraries:
```
Python = 3.10.8   # other versions may also work but are untested
PyTorch = 2.0.1   # other versions may also work but are untested
numpy = 1.25.0    # other versions may also work but are untested
pytube
scipy
pandas
pyautogui
pillow
pycocotools
matplotlib = 3.7.1
tqdm
```

After all of this, if you want to use Text-to-Motion to generate motion sequences, you will need to install the corresponding environment and code found [here](https://github.com/EricGuo5513/text-to-motion). We recommend installing Text-to-Motion in a separate environment to avoid dependency conflicts.

### Other Installations
We use Blender 3.4 for our human rigging. Other versions may work but are untested. Install Blender [here](https://www.blender.org/download/). Add the Blender installation folder to your environment path if not already added.

We also use Unity for our simulation. Install Unity Hub to manage all relevant Unity installations [here](https://unity.com/download). All relevant Unity packages will install automatically when opening the project "[wheelpose_unity_env](wheelpose_unity_env)". If you can not successfully install the Unity environment, here is a list of required tools and their relevant installation instructions. We recommend following the instructions listed due to outdated documentation:

- [Unity 2021.3](https://unity.com/releases/editor/whats-new/2021.3.3)
- [Perception 1.0.0](https://github.com/Unity-Technologies/com.unity.perception)
  - Installation instructions found [here](https://github.com/Unity-Technologies/com.unity.perception/blob/main/com.unity.perception/Documentation~/SetupSteps.md).
- [SyntheticHumans Release CS3199](https://github.com/Unity-Technologies/com.unity.cv.synthetichumans)
  - Installation instructions found [here](https://github.com/Unity-Technologies/com.unity.cv.synthetichumans/wiki/Synthetic-Humans-Tutorial#step-2-install-packages-and-set-things-up). Skip the Unity Perception installation if you have already done so.

## Posture Modification


### Downloading and Collecting Data
For the purposes of our reserach, we use motion sequences collected from two individual sources: HumanML3D and Text-to-Motion. HumanML3D motions can be downloaded [here](https://github.com/EricGuo5513/HumanML3D).

Text-to-Motion is a fully generative deep learning model to generate natural 3D human motions from text. We use Text-to-Motion to generate equivalent motions to HumanML3D by describing the motion we see in HumanML3D into Text-to-Motion.

Both models should output ```.npy``` files of XYZ positions of different joints. Other motion sequence sources can be used including different generative models and dataset. All data must just be converted to the required joint definitions found below.

All inputted data should be in the orientation where +X is in the direction of the face and Y is parallel to the spine of the human.
```
0: "pelvis",
1: "lhip",
2: "rhip",
3: "spine",
4: "rknee",
5: "lknee",
6: "chest",
7: "rankle",
8: "lankle",
9: "upperchest",
10: "lbigtoe",
11: "rbigtoe",
12: "headbase",
13: "lclavicle",
14:"rclavicle",
15: "headtop",
16: "lshoulder",
17: "rshoulder",
18: "lelbow",
19: "relbow",
20: "lwrist",
21: "rwrist",
22: "neckbase" # optional inclusion. Slight modification to code might be necessary if predefining a neckbase
```

### Posture Modification and Conversion
Corresponding motion sequences are processed and converted under the following steps
- Compute any missing joints (neckbase)
- 5Hz low pass filter on joint positions
- Put legs into a sitting position to model users in wheelchairs and add some rotational noise to simulate natural movements within a wheelchair
- Convert XYZ to rotational data around each joint
- Export in the form of a ```.json``` file and ```.npy``` files

All posture modification is performed by running the following:
```
python ./src/posture_modification/pose_conversion.py -i DIRECTORY_OF_INPUT_NPY -o OUTPUT_DIRECTORY
```

Generated files can be visualized by running the following:
```
python ./src/posture_modification/pose_anim_vis.py -i INPUT_NPY -o OUTPUT_GIF
```

### FBX Conversion
In order to convert rotational files to FBX files, we use Blender and a custom Blender script. The conversion can be run through the following:

```
blender ./src/blender_fbx_conversion/Scripting.blend --background --python ./src/blender_fbx_conversion/blender_gen.py -- -i INPUT_DIRECTORY -o OUTPUT_DIRECTORY
```

### AnimationClip Conversion
Unity can directly read FBX animations but SyntheticHumans requires the use of Humanoid AnimationClips to interface with randomizers. We provide a fully free conversion layer using macros to convert between FBX to Humanoid AnimationClip using Unity 2021.3. This can be accomplished through the following steps.

- Create a new folder within the assets folder of the Unity environment provided. This should look like this
```
mkdir ./wheelpose_unity_env/Assets/FBX/
```
- Place all FBX files generated from conversion within your created folder
- Open the Unity project and open the ```./wheelpose_unity_env/Assets/Scenes/ExtractAnim.unity``` scene.
- Open your created folder in the file explorer and click on the first FBX file.
- Select the rig tab.
- Run the following and follow the instructions to set up a macro. An image is provided to help with the setup process. The macro will continue running and can be stopped at any time with ```ctrl+c``` in the terminal.
  1. Right click the file explorer taskbar
  2. Right click the inspector taskbar
  3. Right click the Hierarchy tree
```
python ./src/posture_modification/fbx_to_anim_macro.py
```
![macro_setup](docs/macro_setup.png)

- Enter your corresponding input and output path in the ```FBXToAnim``` component on the ```Extract``` GameObject.
- Run the scene to convert all FBX models into ```.anim``` files.

## Background Images

Background images can be downloaded separately.
- Unity SynthHomes is found [here](https://github.com/Unity-Technologies/SyntheticHomes).
- BG-20K is founder [here](https://github.com/JizhiziLi/GFM).

## Unity Simulation
More information on the Unity simulation can be found [here](wheelpose_unity_env/README.md)

## YouTube Annotation Tool
More information on the YouTube annotation tool can be found [here](src/annotation_tools/README.md)

## Dataset Cleaning Tools
We also provide a set of Python scripts to help manage generated SOLO data from Unity and COCO data for training found in ```./src/dataset_organization/```.

Merge COCO datasets:
```
python ./src/dataset_organization/merge_coco.py -b BASE_COCO_JSON -j COCO_JSON_TO_ADD
```

Sample a portion of an existing COCO dataset:
```
python ./src/dataset_organization/sample_coco.py -i INPUT_COCO_JSON -o OUTPUT_COCO_JSON -n N_IMAGES_TO_SAMPLE -min MINIMUM_FRAME_NUMBER -max MAXIMUM_FRAME_NUMBER
```

Convert Unity SOLO to COCO:
```
python ./src/dataset_organization/solo_to_coco.py -i INPUT_SOLO_DIR -o OUTPUT_COCO_PATH
```

## Detectron2
We provide a set of basic tools to train and test Detectron2.


### Training Detectron2
```
CUDA_VISIBLE_DEVICES=SELECTED_CUDA_GPUS python3 ./src/detectron2_scripts/basic_train.py -jp TRAINING_JSON_PATH -ip TRAINING_IMAGE_DIR -o OUTPUT_FILE_NAME -e N_EPOCHS -bs BATCH_SIZE -ngpu N_GPUS -lr LEARNING_RATE -s-ann STEP ANNEAL -seed SEED

```

### Testing Detectron2
We compute a set of custom keypoint metrics including PDJ, PDJPE, OKS50, OKS75, along with the default AP metrics from Detectron2.

```
CUDA_VISIBLE_DEVICES=SELECTED_CUDA_GPU python src/detectron2_scripts/validation_metrics.py -i GENERATED_DIR_FROM_TRAINING -tjp TESTING_JSON -tdp TESTING_IMAGE_DIR -ni N_IMAGES_TO_VISUALIZE
```

## License
WheelPose is licensed under the MIT License. See [LICENSE](LICENSE) for the full license information.
