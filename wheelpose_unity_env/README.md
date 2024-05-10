# WheelPose Unity Environment

![unity_version](https://img.shields.io/badge/unity-2021.3.19f1-green)

This repository provides the WheelPose Unity environment built in HDRP as an extension of [PeopleSansPeople](https://unity-technologies.github.io/PeopleSansPeople/). Our project includes the custom randomizers developed in PeopleSansPeople updated to **Unity `2021.3.19f1`** for support for **Unity Perception `1.0.0`** and **Unity Synthetic Humans `CS3199`**. Other version may work but are currently untested. We provide the full functionality of WheelPose explained in our paper. All 3rd party assets have been replaced with placeholder GameObjects which can easily be phased out. We also provide the following assets:
- 100 HumanML3D converted animations.
- 100 HumanML3D converted animations with increased wheelchair-human clipping.
- 100 Text-to-Motion converted animations.
- 90 Text-to-Motion human-evaluation filtered animations.
- 4 Unity-branded assets with different clothing colour but the same appearance.
- 529 Unity groceries textures.

We hope this Unity environment can enable future researchers to continue experimenting with human-centric synthetic data focused on mobility-assistive technologies by lowering the barrier to entry for synthetic data generation. Assets, randomizers, and labelers can all be easily removed, added, or modified as needed as long as their properties match those of example assets to create an easily accessible simulation environment.

## Getting Started
- All necessary packages come preinstalled with this project.
- Generated synthetic data will include a set of images, and annotations with bounding box, keypoint labels, and occlusion labeling.

1. Open the project and open `Edit > Project Settings`.

![Project Settings](./docs/Project_Settings.png)

2. In `Project Settings` search for `Lit Shader Mode` and set it to `Both`.

![Lit Shader Mode](./docs/Project_Settings_Lit_Shader_Mode.png)

3. In `Project Settings` search for `Motion Blur` and disable it.

![Motion Blur](./docs/Project_Settings_Motion_Blur.png)

4. In `Project Settings` search for `Graphics > HDRP Global Settings` and click + on the list under `Diffusion Profile Assets`. Note that your list might contain more items which is OK.

![Diffusion Profile Assets](./docs/skin-profile.png)

5. In the opened window, enable the display of non-project assets to see the skin profile included in the Synthetic Humans package. Then select the profile named `Skin.asset` located in the SyntheticHumans package at `Packages/com.unity.cv.synthetichumans/Resources/DiffusionProfiles/Skin.asset`.

## Running the Simulation
We provide two scenes for running simulations.
- `PSPHumanWheelchairScene.unity` consists of the default PSP human models in wheelchairs
- `SyntheticHumansWheelchairScene.unity` consists of SyntheticHumans human models in wheelchairs

Upon opening either scene, you can start running the simulation and expect randomly generated images with annotations like the ones shown in the example below. Ensure that you stay on the **game view** to capture images properly.

![Example Data Collection](./docs/example_data_collection.png)

We provide two different GameObjects present in the scene as well, `Texture Wall` and `Sprite Wall`. Enable each accordingly to use either the Unity default textures or user-defined sprites from the `SpriteRnadomizer`.

Wheelchairs or other mobility assistive technologies can be added in the `WheelchairGenerationBase.prefab` and `PSP_Wheelchair.prefab` prefabs.

All domain randomization parameters can be found and modified in `Hierarchy > Scene > Simulation Scenario > Fixed Length Scenario`. All labeler parameters can be found and modified in `Hierarchy > Scene > Main Camera > Perception Camera`.

## Accessing Data
After you finish running your simulation, you can locate the folder where the dataset is stored under `Hierarchy > Scene > Main Camera > Perception Camera` under the `Latest Generated Dataset` in the inspector tab. It will also be printed to the console in the Unity editor. SOLO can be converted through our dataset organization tools with instructions found in `../README.md`.

## More Information
Additional information on the Unity environment may be found in the PeopleSansPeople documentation [here](https://github.com/Unity-Technologies/PeopleSansPeople/tree/main/peoplesanspeople_unity_env).

## Domain Randomization Information

A brief description of domain randomization parameters follows:

**BackgroundObjectPlacementRandomizer.** Randomly spawns background and occluder objects within a user-defined 3D volume. Separation distance can be set to dictate the proximity of objects from each other. Poisson-Disk sampling is used to randomly place objects sourced from a set of primitive 3D game objects (cubes, cylinders, spheres, etc.) from Unity Perception in a given area.

**BackgroundOccluderScaleRandomizer**. Randomizes the scale of the background and occluder objects.

**RotationRandomizer.** Randomizes the 3D rotation of background and occluder objects.

**ForegroundObjectPlacementRandomizer.** Similar to *BackgroundObjectPlacementRandomizer*. Randomly spawns foreground objects selected from the default set of PSP models affixed in wheelchair models.

**ForegroundScaleRandomizer.** Similar to *BackgroundOccluderScaleRandomizer.* Randomizes the scale of foreground objects.

**TextureRandomizer.** Randomizes the texture of predefined objects provided as a JPEG or PNG. We used the set of example textures from Unity Perception which are applied to the background and occluder objects as well as to the background wall when no specific background is set.

**HueOffsetRandomizer.** Randomizes the hue offset applied to textures on the object. Applied to background and occluder objects as well as to the background wall when no specific background is set.

**SpriteRandomizer.** Randomizes the background wall. Used as an alternative to the *TextureRandomizer* when images should not be stretched to fill a canvas.

**HumanGenerationRandomizer.** Randomizes the age, sex, ethnicity, height, weight, and clothing of spawned human assets. Humans are spawned in batches called pools which are periodically regenerated through the simulation process. All humans are spawned within a predefined base which contains the wheelchair model used. All textures and models used are sourced directly from SyntheticHumans.

**NonTagAnimationRandomizer.** Randomizes the pose applied to a character. The pose is a randomly selected frame from a randomly selected *AnimationClip* taken from a universal pool of *AnimationClips*. Provides a custom alternative to the Unity Perception AnimationRandomizer for randomizing animations taken from a single pool.

**TransformPlacementRandomizer.** Randomizes the position, rotation, and size of generated SyntheticHumans. Rotations around the *X,Z*-axis are limited to better represent real world data where users are rarely seen in such orientations.

**SunAngleRandomizer.** Randomizes a directional light's intensity, elevation, and orientation to mimic the lighting effects of the Sun.

**LightRandomizer.** Randomizes a light's intensity and color (RGBA). Also enables the randomization of a light's on/off state.

**LightPositionRotationRandomizer.** Randomizes a light's global position and rotation in the scene.

**CameraRandomizer.** Randomizes the extrinsic parameters of a camera including its global position and rotation. Enables the randomization of intrinsic camera parameters including field of view and focal length to better mimic a physical camera. Adds camera bloom and lens blur around objects that are out of focus to capture more diverse perspectives of the scene.

**PostProcessVolumeRandomizer.** Randomizes select post processing effects including vignette, exposure, white balance, depth of field, and color adjustments.
