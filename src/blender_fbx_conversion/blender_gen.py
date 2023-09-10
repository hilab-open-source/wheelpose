import bpy
import numpy as np
import math
import mathutils
import json
import time
import os
import argparse
import sys

def get_parser():
    parser = argparse.ArgumentParser(
        prog="Blender conversion to FBX.",
        description="Converts generated rotational npy files to FBX. Must be run through blender"
    )

    parser.add_argument(
                        "-i",
                        "--input",
                        type=str,
                        help="input folder of .npy posture rotational files",
                        required=True
                        )
    parser.add_argument(
                        "-o",
                        "--output",
                        type=str,
                        help="output folder",
                        required=True
                        )
    return parser

parser = get_parser()
args = parser.parse_args(sys.argv[sys.argv.index("--")+1:]) # blender ignores all arguments after --

IN_FOLDER = args.input
OUT_FOLDER = args.output

print(f"STARTING AT TIME {time.time()}\n")

IN_FOLDER = os.path.abspath(IN_FOLDER)
OUT_FOLDER = os.path.abspath(OUT_FOLDER)

current_scene = bpy.context.scene
action_groups = bpy.context.object.animation_data.action.groups
ob = bpy.data.objects["Skeleton"]
all_bones = ob.data.bones
pose_bones = ob.pose.bones
action_groups = bpy.context.object.animation_data.action.groups

print(all_bones.keys())


FINGER_BONES = ["Index", "Middle",
                "Ring", "Pinky"]

IDENTITY_Q = [1,0,0,0]
test_id = 2
for i, file_num in enumerate(os.listdir(IN_FOLDER)):
#    if i<test_id:
#        continue
    print(file_num)
    IN_PATH = os.path.join(IN_FOLDER, file_num, 'local_rotations.json')
    OUT_PATH = os.path.join(OUT_FOLDER, f"{file_num}_edited.fbx")

    with open(IN_PATH, "r") as f:
        global_rotations = json.load(f)

    print(global_rotations.keys())
    n_frames = len(list(global_rotations.values())[0])
    current_scene.frame_end = n_frames

    for b_tuple in all_bones.items():
        bone_name, bone = b_tuple
        # goes through the hands and given rotation bones
        if "Hand" in bone.name:
#            d = [IDENTITY_Q for _ in range(n_frames)]
            d =[]
            bone.use_inherit_rotation = True
        elif "Thumb" in bone.name:
            d = [[-1, .5,.5,-.6] for _ in range(n_frames)]
        elif any([substring in str(bone.name) for substring in FINGER_BONES]):
            d = [[-.873, 0,0, -.483] for _ in range(n_frames)]
        elif bone_name in global_rotations.keys():
            d = global_rotations[bone_name]
        elif bone_name not in global_rotations.keys():
            continue

        group = action_groups.get(bone.name)

        print(bone_name, group.channels[0])

        for f, q in enumerate(d):
            frame = f+1
            fcurves = group.channels

            for j, fc in enumerate(fcurves[3:7]): #quaternions: w,x,y,z
                # remove keyframe points already at edit_frame
                [fc.keyframe_points.remove(p) for p in fc.keyframe_points
                        if abs(p.co.x - frame) < 0.0001]
#                [fc.keyframe_points.remove(p) for p in fc.keyframe_points if p in fc.keyframe_points]
                # insert a keyframe
                fc.keyframe_points.insert(frame, q[j])

            if "LeftArm" in bone_name or "RightArm" in bone_name:
                bpy.context.scene.frame_set(frame)
                bpy.context.view_layer.update()

                left_pose = bpy.context.object.pose.bones["LeftArm"]
                right_pose = bpy.context.object.pose.bones["RightArm"]

                dir_str = bone_name.replace("Arm","")

                new_q = []

                group = bpy.context.object.animation_data.action.groups.get(bone.name)
                fcurves = group.channels
                if dir_str == "Left":
                    orientation= left_pose.vector
                    # print(orientation)
                    orientation_check = orientation[0] > 0

                    original_q = left_pose.rotation_quaternion
                    q = mathutils.Quaternion([0, 1, 0, 0])
                    if orientation_check:
                        new_q = original_q @ q
                    elif orientation[0] < 8:
                        new_q = original_q @ q

                else:
                    orientation= right_pose.vector
                    # print(orientation)
                    orientation_check = orientation[0] < 0

                    original_q = right_pose.rotation_quaternion
                    q = mathutils.Quaternion([0, 1, 0, 0])
                    if orientation_check:
                        new_q = original_q @ q
                    elif orientation[0] < 8:
                        new_q = original_q @ q

                if new_q:
                    for j, fc in enumerate(fcurves[3:7]): #quaternions: w,x,y,z
                        # remove keyframe points already at edit_frame
                        [fc.keyframe_points.remove(p) for p in fc.keyframe_points
                                if abs(p.co.x - frame) < 0.0001]
                        # insert a keyframe
                        fc.keyframe_points.insert(frame, new_q[j])

    # if i==test_id:
    #     break

    #Export File
    print('Starting Export ...')
    bpy.ops.export_scene.fbx(
    filepath=OUT_PATH,
    check_existing=False,
    apply_unit_scale=False,
    use_space_transform=False,
    object_types={'ARMATURE'},
    bake_anim_use_nla_strips=False,
    bake_anim_use_all_actions=False,
    axis_forward='Z',
    axis_up='Z'
    )



print('ENDING\n')
