import numpy as np
import copy
import yaml
import os
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

from math_helper import *
from constants import *

def get_bone_local_axes(b2r):
    # x is always in the direction of the bone
    # y is in the axis of left right rotations in unity
    # z is in the axis of front back rotations

    # takes a bone to reference dictionary and returns a set of unity axes that correspond to their orientation

    axes = {}

    #compute the normal vector of the human (ie direction they face)
    # better be facing the right direction in line with the torso!

    # root /pelvis
    lhip = b2r[(0,1)]
    spine = b2r[(0,3)]
    pelv_y = np.cross(lhip, spine)
    pelv_z = np.cross(lhip, pelv_y)

    pelv_axes = get_axes(lhip, pelv_y, pelv_z)
    axes[(0,1)] = pelv_axes
    axes[(0,2)] = pelv_axes

    lcollar = b2r[(22, 13)]
    upperchest = b2r[(9, 22)]
    upperchesty = np.cross(lcollar, upperchest)
    upperchest_z = np.cross(upperchest, upperchesty)

    axes[(9, 22)] = get_axes(upperchest, upperchesty, upperchest_z)
    axes[(22, 13)] = axes[(9, 22)]
    axes[(22, 14)] = axes[(9, 22)]
    # subdivide any potential rotations down the spine
    spiney = (pelv_y + upperchesty)/4
    spinez = np.cross(spine, spiney)
    axes[(0, 3)] = get_axes(spine, spiney, spinez)

    spine2 = b2r[(3,6)]
    spine2y = (pelv_y + upperchesty)/2
    spine2z = np.cross(spine2, spiney)
    axes[(3,6)] = get_axes(spine2, spine2y, spine2z)

    chest = b2r[(6,9)]
    chesty = (pelv_y + upperchesty) / (4/3)
    chestz = np.cross(chest, chesty)
    axes[(6,9)] = get_axes(chest, chesty, chestz)

    neck = b2r[(22, 12)]
    neckz = np.cross(neck, upperchesty)
    necky = np.cross(neck, neckz)
    axes[(22,12)] = get_axes(neck, necky, neckz)

    head = b2r[(12, 15)]
    headz = np.cross(head, upperchesty)
    heady = np.cross(head, headz)
    axes[(12, 15)] = get_axes(head, heady, headz)

    # shoulders are kinda finnicky
    lshoulder = b2r[(13, 16)]
    lshoulderz = np.cross(lshoulder, upperchesty)
    lshouldery = np.cross(lshoulder, lshoulderz)
    axes[13, 16] = get_axes(lshoulder, lshouldery, lshoulderz)

    rshoulder = b2r[(14, 17)]
    rshoulderz = np.cross(rshoulder, upperchesty)
    rshouldery = np.cross(rshoulder, rshoulderz)
    axes[14, 17] = get_axes(rshoulder, rshouldery, rshoulderz)

    lupperarm = b2r[(16, 18)]
    lforearm = b2r[(18, 20)]

    larm_cross = np.cross(lupperarm, lforearm)
    lupperarmy = np.cross(lupperarm, larm_cross)
    axes[(16, 18)] = get_axes(lupperarm, -lupperarmy, -larm_cross)

    lforearmz = np.cross(lforearm, larm_cross)
    axes[(18, 20)] = get_axes(lforearm, -larm_cross, -lforearmz)
    # axes[(19, 21)] = axes[(17,19)]

    rupperarm = b2r[(17, 19)]
    rforearm = b2r[(19, 21)]

    rarm_cross = np.cross(rupperarm, rforearm)
    rupperarmy = np.cross(rupperarm, rarm_cross)
    axes[(17, 19)] = get_axes(rupperarm, rupperarmy, rarm_cross)

    rforearmz = np.cross(rforearm, rarm_cross)
    axes[(19, 21)] = get_axes(rforearm, -rarm_cross, -rforearmz)
    # axes[(18, 20)] = axes[(16,18)]

    lthigh = b2r[(1, 4)]
    lshin = b2r[(4, 7)]
    lfoot = b2r[(7, 10)]

    lleg_cross = np.cross(lthigh, lshin)

    lthighy = np.cross(lthigh, lleg_cross)
    axes[(1,4)] = get_axes(lthigh, -lthighy, lleg_cross)

    lshinz = -np.cross(lshin, lleg_cross)
    axes[(4, 7)] = get_axes(lshin, lleg_cross, lshinz)
    # axes[(5, 8)] = axes[(2,5)]

    lfooty = np.cross(lfoot, lshin)
    lfootz = np.cross(lfoot, lfooty)
    axes[(7, 10)] = get_axes(lfoot, lfooty, lfootz)

    rthigh = b2r[(2, 5)]
    rshin = b2r[(5, 8)]
    rfoot = b2r[(8, 11)]
    rleg_cross = np.cross(rthigh, rshin)

    rthighy = np.cross(rthigh, rleg_cross)
    axes[(2,5)] = get_axes(rthigh, -rthighy, -rleg_cross)

    rshinz = np.cross(rshin, rleg_cross)
    axes[(5, 8)] = get_axes(rshin, -rleg_cross, -rshinz)
    # axes[(4, 7)] = axes[(1,4)]

    rfooty = np.cross(rfoot, rshin)
    rfootz = np.cross(rfoot, rfooty)
    axes[(8, 11)] = get_axes(rfoot, rfooty, rfootz)

    axes[(1,4)] = [[0,0,1], [0,-1,0], [-1,0,0]] #"Left Upper Leg"
    axes[(4,7)] = [[0,0,1], [0,-1,0], [-1,0,0]] #"Left Lower Leg"
    axes[(7,10)] = [[0,0,-1], [0,1,0], [-1,0,0]] #left foot

    axes[(2,5)] = [[0,0,1], [0,1,0], [-1,0,0]] #"Right Upper Leg"
    axes[(5,8)] = [[0,0,1], [0,1,0], [-1,0,0]] #"Right Lower Leg"
    axes[(8,11)] = [[0,0,-1], [0,-1,0], [-1,0,0]] #right foot

    axes[(14, 17)] = [[1,0,0], [0,1,0], [0,0,1]] #Right Shoulder
    axes[(17,19)] = [[0,1,0], [0,0,1], [1,0,0]] #Right UpperArm
    axes[(19,21)] = [[0,-1,0], [0,0,-1], [-1,0,0]] #Right Arm

    axes[(13, 16)] = [[1,0,0], [0,-1,0], [0,0,1]] #Left Shoulder
    axes[(16,18)] = [[0,-1,0], [0,0,-1], [-1,0,0]] #Left UpperArm
    axes[(18,20)] = [[0,-1,0], [0,0,-1], [-1,0,0]] #Left Arm

    axes[(22,12)] = [[1,0,0], [0,1,0], [0,0,1]] #Neck
    axes[(12,15)] = [[1,0,0], [0,1,0], [0,0,1]] #Head

    axes[(0,3)] = [[1,0,0], [0,-1,0], [0,0,1]] #Spine1
    axes[(3,6)] = [[1,0,0], [0,-1,0], [0,0,1]] #Spine2
    return axes

def orient_data(data, fin_orient):
    # orients the data so that the body  points in the direction of orient
    neck = data[:,15,:] - data[:,12,:]
    neck /= np.linalg.norm(neck, axis=1)[:,None]
    neck = np.mean(neck, axis=0)
    if neck[2] > 0:
        lhip = data[:,1,:] - data[:,0,:]
        rhip = data[:,2,:] - data[:,0,:]

        # gets the current average orientation of the user
        curr_orient = np.cross(rhip, lhip, axis=1)
        curr_orient /= np.linalg.norm(curr_orient, axis=1)[:,None]
        curr_orient = np.mean(curr_orient, axis=0)

        q = calc_rot_q(np.array([curr_orient]), np.array([fin_orient]))[0]
        # rounds the rotation to trasform the current value to desired orientation for a rough orient
        r = R.from_quat(wxyz_to_xyzw(q)).as_euler("xyz", degrees=True)
        q_euler = [base_round(val, 90) for val in r] # rounds to the nearest 90 degrees

        r = R.from_euler("xyz", q_euler, degrees=True)
        # r = R.from(wxyz_to_xyzw(q))
        for i in range(len(data[0,:,0])):
            data[:,i,:] = r.apply(data[:,i,:])

        r= R.from_euler("x", 90, degrees=True)
        for i in range(len(data[0,:,0])):
            data[:,i,:] = r.apply(data[:,i,:])

        return data

    else:
        data[:, :, [1,2]] = data[:, :, [2,1]]
        data = np.append(data, np.mean(data[:, [13, 14], :], axis=1).reshape(len(data),1,3), axis=1)

        # orienting into blender

        # need some orientation code cause thanks eric
        r= R.from_euler("z", 90, degrees=True)
        for i in range(len(data[0,:,0])):
            data[:,i,:] = r.apply(data[:,i,:])

        return data


def global_rot_reconstruct(bone_ref, rot, n_frames, bone_lens=None):
    # recreates the data from global rotations and reference
    reconstructed_data =[]
    for frame in range(n_frames):
        joint_pos_dict = {0: np.array([0,0,0])}

        for bone, ref in bone_ref.items():
            scale = 1 if bone_lens == None else bone_lens[bone]

            q = rot[bone][frame]

            # q = [1,0,0,0]
            q = q[1], q[2], q[3], q[0] # scipy takes it in xyzw

            r = R.from_quat(q)
            v = r.apply(ref) * np.abs(scale) # bone reference is initially in unit vector form
            joint_pos_dict[bone[1]] = joint_pos_dict[bone[0]] + v

        k = list(joint_pos_dict.keys())
        k.sort()
        joint_pos_dict = {i: joint_pos_dict[i] for i in k}
        reconstructed_data.append(np.array(list(joint_pos_dict.values())))

    reconstructed_data = np.array(reconstructed_data)

    return reconstructed_data

def graph_eulers(d, title):
    plt.figure(figsize=(12,12))
    plt.plot(d[:,0], label="x")
    plt.plot(d[:,1], label="y")
    plt.plot(d[:,2], label="z")
    plt.xlabel("idx")
    plt.ylabel("Angle (Radians)")
    plt.title(title)
    plt.legend()
    plt.show()
