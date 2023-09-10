E = 1e-15
GLOBAL_AXIS = [[1,0,0],
                [0,1,0],
                [0,0,1]]

AX_TO_IDX = {"x": 0,
             "y": 1,
             "z": 2}

QUAT_TO_IDX = {
               "w":0,
               "x":1,
               "y":2,
               "z":3
               }
# chains of joints to create different parts of body
KEYPOINT_NAMES = {
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
    22: "neckbase"
}

# set of references mapping a bone to what their rotations will be in respect to
TPOSE_REF = {
    (0, 2): (0.0, -1.0, 0.0), 
    (2, 5): (0.0, 0.0, -1.0),
    (5, 8): (0.0, 0.0, -1.0),
    (8, 11): (1.0, 0.0, 0.0),
    (0, 1): (0.0, 1.0, 0.0),
    (1, 4): (0.0, 0.0, -1.0),
    (4, 7): (0.0, 0.0, -1.0),
    (7, 10): (1.0, 0.0, 0.0),
    (0, 3): (0.0, 0.0, 1.0),
    (3, 6): (0.0, 0.0, 1.0),
    (6, 9): (0.0, 0.0, 1.0),
    (9, 22): (0.0, 0.0, 1.0),
    (22, 12): (0.0, 0.0, 1.0),
    (12, 15): (0.0, 0.0, 1.0),
    (22, 14): (0.0, -1.0, 0.0),
    (14, 17): (0.0, -1.0, 0.0),
    (17, 19): (0.0, -1.0, 0.0),
    (19, 21): (0.0, -1.0, 0.0),
    (22, 13): (0.0, 1.0, 0.0),
    (13, 16): (.0, 1.0, 0.0),
    (16, 18): (.0, 1.0, 0.0),
    (18, 20): (.0, 1.0, 0.0)
}

UNITY_REF = {
    (0, 2): (1.0, 0.0, 0.0), 
    (2, 5): (0.0, -0.8700574239812991, 0.49295038185909307), 
    (5, 8): (0.0, -0.5679578868558361, -0.8230576156978643), 
    (8, 11): (0.0, -0.8075786430198312, 0.5897598963461725), 
    (0, 1): (-1.0, 0.0, 0.0), 
    (1, 4): (0.0, -0.8700574239812991, 0.49295038185909307), 
    (4, 7): (0.0, -0.5679578868558361, -0.8230576156978643), 
    (7, 10): (0.0, -0.8075786430198312, 0.5897598963461725), 
    (0, 3): (0.0, 1.0, 0.0), 
    (3, 6): (0.0, 1.0, 0.0), 
    (6, 9): (0.0, 1.0, 0.0), 
    (9, 22): (0.0, 1.0, 0.0), 
    (22, 12): (0.0, 1.0, 0.0), 
    (12, 15): (0.0, 1.0, 0.0), 
    (22, 14): (1.0, 0.0, 0.0), 
    (14, 17): (1.0, 0.0, 0.0), 
    (17, 19): (0.6179775280898878, -0.6741573033707866, 0.40449438202247195), 
    (19, 21): (-0.3422226359227673, 0.03338757423636754, 0.9390255253978371), 
    (22, 13): (-1.0, 0.0, 0.0), 
    (13, 16): (-1.0, 0.0, 0.0), 
    (16, 18): (-0.6179775280898878, -0.6741573033707866, 0.40449438202247195), 
    (18, 20): (0.3422226359227673, 0.03338757423636754, 0.9390255253978371)
}

# based off unity mapping
# unity doesnt use traditional keypoints so 
# these are the mappings of bone to the movement with center of rotation at that joint
# ankle and bigtoe are mapped to a Foot quaternion and translation in unity
# pelvis and hips are mapped to a root quaternion and translation in unity
# root is too
BONE_NAMES = {
    (0,1): "Left Root", 
    (0,2): "Right Root", 
    (1,4): "Left Upper Leg",
    (2,5): "Right Upper Leg",
    (0,3): "Spine",
    (3,6): "Chest",
    (6,9): "Sternum",
    (9,22): "Upper Chest",
    (22, 13): "Left Clavicle",
    (22, 14): "Right Clavicle",
    (4,7): "Left Lower Leg",
    (5,8): "Right Lower Leg",
    (7,10): "Left Foot",
    (8,11): "Right Foot",
    (12,15): "Head",
    (13, 16): "Left Shoulder",
    (14, 17): "Right Shoulder",
    (16, 18): "Left Arm",
    (17, 19): "Right Arm",
    (18, 20): "Left Forearm",
    (19, 21): "Right Forearm",
    (22, 12): "Neck"
}

# blender bone mappings
BLENDER_BONE_NAMES = {
    # (0,1): "Left Root", 
    # (0,2): "Right Root", 
    (1,4): "LeftUpperLeg",
    (2,5): "RightUpperLeg",
    (0,3): "Spine1",
    (3,6): "Spine2",
    # (6,9): "Sternum",
    # (9,22): "Upper Chest",
    # (22, 13): "Left Clavicle",
    # (22, 14): "Right Clavicle",
    (4,7): "LeftLeg",
    (5,8): "RightLeg",
    (7,10): "LeftFoot",
    (8,11): "RightFoot",
    (12,15): "Head",
    (13, 16): "LeftShoulder",
    (14, 17): "RightShoulder",
    (16, 18): "LeftUpperArm",
    (17, 19): "RightUpperArm",
    (18, 20): "LeftArm",
    (19, 21): "RightArm",
    (22, 12): "Neck"
}

#UNITY: TWIST IN OUT, twist left-right: X
# down up: y
# front back, in out: z
# stretch : one degree of freedom so just 1 angle

# ASSIGNING DIMENSIONS WHERE X IS ALWAYS THE MAJOR AXIS WHEN IN TPOSE

UNITY_ROT_TO_EULER = {
    "Twist Left-Right": 0,
    "Twist In-Out": 0,
    "Turn Left-Right": 1,
    "Nod Down-Up": 1,
    "Down-Up": 1,
    "Up-Down": 1,
    "In-Out": 1,
    "Front-Back": 2,
    "Tilt Left-Right": 2,
    "Left-Right": 2,
    "Stretch": 1
}