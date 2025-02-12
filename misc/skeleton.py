import numpy as np

def get_flip_indices(num_joints, left_indices, right_indices):
    flip_indices = []
    for i in range(num_joints):
        if i in left_indices:
            flip_indices.append(right_indices[left_indices.index(i)])
        elif i in right_indices:
            flip_indices.append(left_indices[right_indices.index(i)])
        else:
            flip_indices.append(i)
    
    return flip_indices
        
def get_left_right_bones(bones, left_indices, right_indices, flip_indices):
    left_bones = []
    right_bones = []
    for bone in bones:
        if bone[0] in left_indices and bone[1] not in right_indices:
            left_bones.append(bone)

    for left_bone in left_bones:
        right_bone = [flip_indices[left_bone[0]], flip_indices[left_bone[1]]]
        right_bones.append(right_bone)

    return left_bones, right_bones
        
class COCOSkeleton:
    joint_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
        'right_knee', 'left_ankle', 'right_ankle'
    ]
    bones = [
        [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6], [5, 7], [6, 8], [7, 9], [8, 10], [5, 11],
        [6, 12], [11, 13], [12, 14], [13, 15], [14, 16]
    ]
    left_indices = [1, 3, 5, 7, 9, 11, 13, 15]
    right_indices = [2, 4, 6, 8, 10, 12, 14, 16]
        
    num_joints = len(joint_names)
    flip_indices = get_flip_indices(num_joints, left_indices, right_indices)
    left_bones, right_bones = get_left_right_bones(bones, left_indices, right_indices, flip_indices)
    center = 0
            
class SimpleCOCOSkeleton:
    joint_names = [
        'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
        'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    bones = [
        [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [1, 7],
        [2, 8], [7, 9], [8, 10], [9, 11], [10, 12]
    ]
    left_indices = [1, 3, 5, 7, 9, 11]
    right_indices = [2, 4, 6, 8, 10, 12]
        
    num_joints = len(joint_names)
    flip_indices = get_flip_indices(num_joints, left_indices, right_indices)
    left_bones, right_bones = get_left_right_bones(bones, left_indices, right_indices, flip_indices)
    center = 0
    
class MMBodySkeleton:
    joint_names = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", 
        "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", 
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"
    ]
    bones = [
        [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], 
        [9, 13], [9, 14], [12, 13], [12, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], 
        [18, 20], [19, 21]
    ]
    left_indices = [1, 3, 5, 7, 9, 11, 13, 15]
    right_indices = [2, 4, 6, 8, 10, 12, 14, 16]
        
    num_joints = len(joint_names)
    flip_indices = get_flip_indices(num_joints, left_indices, right_indices)
    left_bones, right_bones = get_left_right_bones(bones, left_indices, right_indices, flip_indices)
    center = 0

class MMFiSkeleton:
    joint_names = [
        "pelvis", "left_hip", "left_knee", "left_ankle", "right_hip", "right_knee", "right_ankle", 
        "waist", "neck", "nose", "head", "right_shoulder", "right_elbow", "right_wrist", 
        "left_shoulder", "left_elbow", "left_wrist"
    ]
    bones = [
        [0, 1], [0, 4], [0, 7], [1, 2], [2, 3], [4, 5], [5, 6], [7, 8], [8, 9], [8, 11], [8, 14],
        [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]
    ]
    left_indices = [1, 2, 3, 14, 15, 16]
    right_indices = [4, 5, 6, 11, 12, 13]
        
    num_joints = len(joint_names)
    flip_indices = get_flip_indices(num_joints, left_indices, right_indices)
    left_bones, right_bones = get_left_right_bones(bones, left_indices, right_indices, flip_indices)
    center = 0

class ITOPSkeleton:
    joint_names = [
        "nose", "neck", "right_shoulder", "left_shoulder", "right_elbow", "left_elbow", "right_wrist", 
        "left_wrist", "spine", "right_hip", "left_hip", "right_knee", "left_knee", "right_ankle", 
        "left_ankle"
    ]
    # joint_names = [
    #     'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
    #     'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    # ] 
    bones = [
        [14, 12], [12, 10], [13, 11], [11, 9],  [10, 8], [9, 8], [8, 1], [1, 0], [7, 5], [5, 3], 
        [3, 1], [6, 4], [4, 2], [2, 1]
    ]
    left_indices = [3, 5, 7, 10, 12, 14]
    right_indices = [2, 4, 6, 9, 11, 13]

    num_joints = len(joint_names)
    flip_indices = get_flip_indices(num_joints, left_indices, right_indices)
    left_bones, right_bones = get_left_right_bones(bones, left_indices, right_indices, flip_indices)
    center = 0

class SMPLSkeleton:
    joint_names = [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", 
        "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", 
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", 
        "left_hand", "right_hand"
    ]
    bones = [
        [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11],
        [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21],
        [20, 22], [21, 23]
    ]
    left_indices = [1, 4, 7, 10, 13, 16, 18, 20, 22]
    right_indices = [2, 5, 8, 11, 14, 17, 19, 21, 23]
        
    num_joints = len(joint_names)
    flip_indices = get_flip_indices(num_joints, left_indices, right_indices)
    left_bones, right_bones = get_left_right_bones(bones, left_indices, right_indices, flip_indices)
    center = 0

def coco2simplecoco(joints):
    return joints[..., [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], :]

def mmbody2simplecoco(joints):
    return joints[..., [15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8], :]

def mmfi2simplecoco(joints):
    return joints[..., [9, 14, 11, 15, 12, 16, 13, 1, 4, 2, 5, 3, 6], :]

def itop2simplecoco(joints):
    return joints[..., [0, 3, 2, 5, 4, 7, 6, 10, 9, 12, 11, 14, 13], :]

def simplecoco2smpl(joints):
    def _j(indices):
        return joints[..., indices, :]
    def _len(j0, j1):
        return np.linalg.norm(j0 - j1, axis=-1, keepdims=True)

    neck = ((_j(1) + _j(2)) / 2 + _j(0)) / 2
    hip_center = (_j(7) + _j(8)) / 2 
    d_spine = neck - hip_center
    d_spine = d_spine / _len(hip_center, neck)
    pelvis = hip_center + d_spine * _len(_j(7), _j(8))
    spine2 = (pelvis + neck) / 2
    spine1 = (pelvis + spine2) / 2
    spine3 = neck * 0.2 + spine2 * 0.8
    spine_pseudo = (neck + spine2) / 2
    left_collar = (_j(1) + spine_pseudo) / 2
    right_collar = (_j(2) + spine_pseudo) / 2
    d_lhand = _j(5) - _j(3)
    d_lhand = d_lhand / _len(_j(3), _j(5))
    left_hand = _j(5) + d_lhand * _len(_j(5), _j(3)) * 0.25
    d_rhand = _j(6) - _j(4)
    d_rhand = d_rhand / _len(_j(4), _j(6))
    right_hand = _j(6) + d_rhand * _len(_j(6), _j(4)) * 0.25
    d_lfoot = (_j(2) - _j(1)) @ (_j(11) - _j(9))
    d_lfoot = d_lfoot / np.linalg.norm(d_lfoot, axis=-1, keepdims=True)
    left_foot = _j(11) + d_lfoot * _len(_j(11), _j(9)) * 0.25
    d_rfoot = (_j(1) - _j(2)) @ (_j(10) - _j(12))
    d_rfoot = d_rfoot / np.linalg.norm(d_rfoot, axis=-1, keepdims=True)
    right_foot = _j(10) + d_rfoot * _len(_j(10), _j(12)) * 0.25

    return np.stack([pelvis, _j(7), _j(8), spine1, _j(9), _j(10), spine2, 
                           _j(11), _j(12), spine3, left_foot, right_foot, neck,
                            left_collar, right_collar, _j(0), _j(1), _j(2), _j(3),
                            _j(4), _j(5), _j(6), left_hand, right_hand], axis=-2)

# class MMWaveGraph(Graph):
#     def __init__(self,
#                  layout='simplecoco',
#                  mode='spatial',
#                  max_hop=1,
#                  nx_node=1,
#                  num_filter=3,
#                  init_std=0.02,
#                  init_off=0.04):
        
#         self.max_hop = max_hop
#         self.layout = layout
#         self.mode = mode
#         self.num_filter = num_filter
#         self.init_std = init_std
#         self.init_off = init_off
#         self.nx_node = nx_node

#         assert nx_node == 1 or mode == 'random', "nx_node can be > 1 only if mode is 'random'"
#         assert layout in ['coco', 'simplecoco', 'mmbody', 'mmfi']

#         self.get_layout(layout)
#         self.hop_dis = get_hop_distance(self.num_node, self.inward, max_hop)

#         assert hasattr(self, mode), f'Do Not Exist This Mode: {mode}'
#         self.A = getattr(self, mode)()

#     def get_layout(self, layout):
#         if layout == 'coco':
#             self.num_node = COCOSkeleton.num_joints
#             self.inward = COCOSkeleton.bones
#             self.center = COCOSkeleton.center
#         elif layout == 'simplecoco':
#             self.num_node = SimpleCOCOSkeleton.num_joints
#             self.inward = SimpleCOCOSkeleton.bones
#             self.center = SimpleCOCOSkeleton.center
#         elif layout == 'mmbody':
#             self.num_node = MMBodySkeleton.num_joints
#             self.inward = MMBodySkeleton.bones
#             self.center = MMBodySkeleton.center
#         elif layout == 'mmfi':
#             self.num_node = MMFiSkeleton.num_joints
#             self.inward = MMFiSkeleton.bones
#             self.center = MMFiSkeleton.center
#         else:
#             raise ValueError(f'Invalid Layout: {layout}')

#         self.self_link = [(i, i) for i in range(self.num_node)]
#         self.outward = [(j, i) for i, j in self.inward]
#         self.neighbor = self.inward + self.outward