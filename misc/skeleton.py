from abc import ABC, abstractmethod

class Skeleton(ABC):

    @abstractmethod
    def __init__(self):
        pass

    def __len__(self):
        return len(self.joint_names)
    
    def _get_flip_indices(self):
        flip_indices = []
        for i in range(self.num_joints):
            if i in self.left_indices:
                flip_indices.append(self.right_indices[self.left_indices.index(i)])
            elif i in self.right_indices:
                flip_indices.append(self.left_indices[self.right_indices.index(i)])
            else:
                flip_indices.append(i)
        return flip_indices
    
    def _get_left_right_bones(self):
        left_bones = []
        right_bones = []
        for bone in self.bones:
            if bone[0] in self.left_indices and bone[1] not in self.right_indices:
                left_bones.append(bone)

        for left_bone in left_bones:
            right_bone = [self.flip_indices[left_bone[0]], self.flip_indices[left_bone[1]]]
            right_bones.append(right_bone)

        return left_bones, right_bones
    
class COCOSkeleton(Skeleton):
    def __init__(self):
        super().__init__()
        self.joint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee',
            'right_knee', 'left_ankle', 'right_ankle'
        ]
        self.bones = [
            [0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6], [5, 7], [6, 8], [7, 9], [8, 10], [5, 11],
            [6, 12], [11, 13], [12, 14], [13, 15], [14, 16]
        ]
        self.left_indices = [1, 3, 5, 7, 9, 11, 13, 15]
        self.right_indices = [2, 4, 6, 8, 10, 12, 14, 16]
        self.flip_indices = super()._get_flip_indices()
        self.left_bones, self.right_bones = super()._get_left_right_bones()

class SimpleCOCOSkeleton(Skeleton):
    def __init__(self):
        super().__init__()
        self.joint_names = [
            'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
            'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        self.bones = [
            [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [1, 7],
            [2, 8], [7, 9], [8, 10], [9, 11], [10, 12]
        ]
        self.left_indices = [1, 3, 5, 7, 9, 11]
        self.right_indices = [2, 4, 6, 8, 10, 12]
        self.flip_indices = super()._get_flip_indices()
        self.left_bones, self.right_bones = super()._get_left_right_bones()

class MMBodySkeleton(Skeleton):
    def __init__(self):
        super().__init__()
        self.joint_names = [
            "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle", 
            "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head", 
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"
        ]
        self.joint_names = [
            'nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 
            'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        self.bones = [
            [0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], 
            [9, 13], [9, 14], [12, 13], [12, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], 
            [18, 20], [19, 21]
        ]
        self.left_indices = [1, 3, 5, 7, 9, 11, 13, 15]
        self.right_indices = [2, 4, 6, 8, 10, 12, 14, 16]
        self.flip_indices = super()._get_flip_indices()
        self.left_bones, self.right_bones = super()._get_left_right_bones()

cocoSkeleton = COCOSkeleton()
simpleCocoSkeleton = SimpleCOCOSkeleton()
mmBodySkeleton = MMBodySkeleton()

def coco2simplecoco(joints):
    return joints[..., [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], :]

def mmbody2simplecoco(joints):
    return joints[..., [15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8], :]