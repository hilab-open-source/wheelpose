import numpy as np
import matplotlib.pyplot as plt
import argparse

import os
from tqdm import tqdm

from scipy.spatial.transform import Rotation as R

import copy
import json

import math_helper as m
from constants import *

import pose_helper as h
# DATA SHOULD BE IN THE FORM OF Y IS PARALLEL IN THE DIRECTION OF THE STANDING HUMAN, +X IS IN DIRECTION OF FACE IN TPOSE
# ALL ROTATIONS IN RESPECT TO SOME REFERENCE MODEL
# GLOBAL IS IN REFERENCE TO THE GLOBAL XYZ AND THE BONE REFERENCE MODEL

class Keypoint:
    def __init__(self, name, data):
        self.name = name
        self.data = data

        self.neighbors = []

    def add_neighbor(self, n):
        if n not in self.neighbors:
            self.neighbors.append(n)

    def get_neighbors(self):
        return self.neighbors.copy()

    def reset(self, data):
        self.data = data

class Bone:
    def __init__(self, name, bone_kp, data, ref, local_axis):
        # kp v is the bone vectors
        # ref v is the actual reference vector
        # local_axis, xyz vectors in global position of bones local axis

        self.name = name
        self.bone_kp = bone_kp
        self.data = data
        self.ref = np.array(ref)
        self.local_axis = local_axis
        self.n_frames = len(data)

        # wxyz quaternion
        # global rotations from reference bone
        self.global_rotations = self.calc_global_rotations(self.ref)
        self.local_rotations = self.calc_local_rotations(self.global_rotations)

    def calc_global_rotations(self, r):
        # r is some reference vector
        # calcs the closest rotation quaternion in the reference of global xyz

        ref_vecs = np.array([r for _ in range(len(self.data))])
        return m.calc_rot_q(ref_vecs, self.data)

    def get_global_rotations(self):
        # returns the global rotation in reference to the initialized reference vector
        return self.global_rotations

    def calc_local_rotations(self, global_rotations):
        # computes the given rotations in terms of the local axis
        local_rotations = []
        for q in global_rotations:
            local_rotations.append(m.q_to_local(q, GLOBAL_AXIS, self.local_axis))

        return np.array(local_rotations)

    def get_local_rotations(self, fill=True):
        return self.local_rotations

    def reset(self, data):
        self.data = data
        self.n_frames = len(data)

        self.global_rotations = self.calc_global_rotations(self.ref)
        self.local_rotations = self.calc_local_rotations(self.global_rotations)

class T2MToAnim:
    def __init__(self, data, fps, bone_ref, local_axes, lp_cutoff=-1):
        # expecting already corrected formated data
        # adds a line for the neck bone part
        data = np.append(data, np.mean(data[:, [13, 14], :], axis=1).reshape(len(data),1,3), axis=1)
        # nx23x3 array for n timesteps
        self.n_frames = len(data)
        # initiating bones and keypoints
        self.bones = {}
        self.keypoints = {}
        self.fps = fps
        self.lp_cutoff = lp_cutoff

        if lp_cutoff != -1:
            self.apply_lpf(lp_cutoff)

        # initiating bones
        for bone, ref in bone_ref.items():
            #keypoint initiation
            for kp in bone:
                if kp in self.keypoints:
                    continue

                self.keypoints[kp] = Keypoint(KEYPOINT_NAMES[kp], data[:,kp,:])
            self.keypoints[bone[0]].add_neighbor(bone[1])# creates a directed graph

            if bone in self.bones:
                continue
            bone_v = m.get_vector(data[:,bone[0],:], data[:, bone[1],:], normalize=True)
            self.bones[bone] = Bone(BONE_NAMES[bone], bone, bone_v, ref, local_axes[bone])

        self.roots = [] # not actually necessary just for easier reading
        self.set_roots()

        # self.calc_all_local_rot()

    def gen_synthetic_sitting_legs(self):
        # lhip = self.bones[(0,1)].data
        # rhip = self.bones[(0,2)].data

        # thigh = np.cross(rhip, lhip, axis=1)
        # thigh /= np.linalg.norm(thigh, axis=1)[:,None]
        # shin = np.cross(lhip-rhip, thigh, axis=1)
        # shin /= np.linalg.norm(thigh, axis=1)[:,None]

        # thigh = np.mean(thigh, axis=0)
        # shin = np.mean(shin, axis=0)

        # thigh[0] = np.abs(thigh[0])

        thigh = [1.5,0,0]
        shin = [0, 0,-0.5]

        thigh = [1.5,0,1.15] # slight rotation up of about 35 degrees from before
        shin = [0, 0,-0.5]



        ranges = ((-10, 10), (-10, 10), (-10,10))
        for v, b in zip((thigh, thigh, shin, shin, thigh, thigh), ((1,4), (2,5), (4,7), (5,8), (7,10), (8,11))):
            d = m.rand_time_var_vectors(v, ranges, self.n_frames)
            d *= np.linalg.norm(self.bones[b].data, axis=1)[:,None] # scale back up

            for i in range(3):
                d[:,i] = m.lp_filter(d[:,i], self.fps, self.lp_cutoff)
            self.bones[b].reset(d) # reset the bone data

            self.keypoints[b[1]].reset(self.keypoints[b[0]].data + d)

    def apply_lpf(self, cutoff):
        s = data.shape
        for i in range(s[1]):
            for j in range(s[2]): # apply lpf
                data[:,i,j] = m.lp_filter(data[:,i,j], self.fps, self.lp_cutoff)

    def set_roots(self):
        # iterates through all bones to find the starting location of all bones

        self.roots = list(self.keypoints.keys())

        for bone in self.bones.keys():
            if bone[1] in self.roots:
                # means the keypoint has an edge that points into it
                self.roots.remove(bone[1])

    def get_bone_global_rot(self):
        rots = {}

        for bone_kp, bone in self.bones.items():
            rots[bone_kp] = bone.get_global_rotations()

        return rots

    def get_bone_local_rot(self, euler=""):
        rots = {}
        for bone_kp, bone in self.bones.items():
            rots[bone_kp] = bone.get_local_rotations()

            if euler != "":
                eulers = []
                for i in range(len(rots[bone_kp])):
                    eulers.append(R.from_quat(m.wxyz_to_xyzw(rots[bone_kp][i])).as_euler(euler))

                rots[bone_kp] = np.array(eulers)
        return rots

def get_parser():
    parser = argparse.ArgumentParser(
        prog="Posture Modifcation",
        description="Converts inputted XYZ motion data to rotational data from a T-Pose reference."
    )

    parser.add_argument(
                        "-i",
                        "--input",
                        type=str,
                        help="input directory of .npz XYZ joint motion files",
                        required=True
                        )
    parser.add_argument(
                        "-o",
                        "--output",
                        type=str,
                        help="output directory",
                        required=True
                        )
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    IN_PATH = os.path.abspath(args.input)
    OUT_FOLDER = os.path.abspath(args.output)

    for file in tqdm(os.listdir(IN_PATH)):
        file_num = file.rsplit(".npy")[0]
        OUT_PATH = os.path.join(OUT_FOLDER, file_num)
        if not os.path.exists(OUT_PATH):
            # Create a new directory because it does not exist
            os.makedirs(OUT_PATH)

        JOINTS_PATH = os.path.join(IN_PATH, file)

        reference = TPOSE_REF

        data = np.load(JOINTS_PATH)
        data = np.append(data, np.mean(data[:, [13, 14], :], axis=1).reshape(len(data),1,3), axis=1)
        data = h.orient_data(data, [1,0,0]) # orients data so it faces 1,0,0

        local_axes = h.get_bone_local_axes(reference)

        try:
            T2MToAnimTest = T2MToAnim(data, 20, reference, local_axes, 5)
            T2MToAnimTest.gen_synthetic_sitting_legs()
            global_rotations = T2MToAnimTest.get_bone_global_rot()
            local_rotations = T2MToAnimTest.get_bone_local_rot()
        except:
            print("input vector mismatch: ", file_num)

        np.save(os.path.join(OUT_PATH, "local_rotations"), local_rotations)

        local_rot_json = {}
        for b, d in local_rotations.items():
            if b in BLENDER_BONE_NAMES:
                local_rot_json[BLENDER_BONE_NAMES[b]] = d.tolist()

        with open(os.path.join(OUT_PATH, "local_rotations.json"), "w") as f:
            json.dump(local_rot_json, f)

        #Optional Gif Animation Code
        # bone_lens = {bone: np.mean(np.linalg.norm(data[:, bone[1],:] - data[:, bone[0],:], axis=1)) for bone in reference.keys()}
        # reconstructed_data = h.global_rot_reconstruct(reference, global_rotations, len(data), bone_lens=bone_lens)
        # from t2m_anim_vis import plot_3d_motion
        # plot_3d_motion(os.path.join(OUT_PATH, "reconstructed_anim.gif"), reconstructed_data, title=file_num, fps=20, radius=1.5)
