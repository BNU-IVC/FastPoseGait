from data import transform as base_transform
import numpy as np
import math
from utils import is_list, is_dict, get_valid_args
import torchvision.transforms as T

class NoOperation():
    def __call__(self, x):
        return x


class RandomSelectSequence(object):
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        try:
            start = np.random.randint(0, data.shape[0] - self.sequence_length)
        except ValueError:
            print(data.shape[0])
            raise ValueError
        end = start + self.sequence_length
        return data[start:end]


class SelectSequenceCenter(object):
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, data):
        try:
            start = int((data.shape[0]/2) - (self.sequence_length / 2))
        except ValueError:
            print(data.shape[0])
            raise ValueError
        end = start + self.sequence_length
        return data[start:end]


class MirrorPoses(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, data):
        if np.random.random() <= self.probability:
            center = np.mean(data[:, :, 0], axis=1, keepdims=True)
            data[:, :, 0] = center - data[:, :, 0] + center

        return data


class NormalizeEmpty(object):
    """
    Normliza Empty Joint
    """
    def __call__(self, data):
        frames, joints = np.where(data[:, :, 0] == 0)
        for frame, joint in zip(frames, joints):
            center_of_gravity = np.mean(data[frame], axis=0)
            data[frame, joint, 0] = center_of_gravity[0]
            data[frame, joint, 1] = center_of_gravity[1]
            data[frame, joint, 2] = 0
        return data


class RandomMove(object):
    """
    Move: add Random Movement to each joint
    """
    def __init__(self,random_r =[4,1]):
        self.random_r = random_r
    def __call__(self, data):
        noise = np.zeros(3)
        noise[0] = np.random.uniform(-self.random_r[0], self.random_r[0])
        noise[1] = np.random.uniform(-self.random_r[1], self.random_r[1])
        data += np.tile(noise,(data.shape[0], data.shape[1], 1))
        return data


class PointNoise(object):
    """
    Add Gaussian noise to pose points
    std: standard deviation
    """
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, data):
        noise = np.random.normal(0, self.std, data.shape).astype(np.float32)
        return data + noise


class FlipSequence(object):
    """
    Temporal Fliping
    """
    def __init__(self, probability=0.5):
        self.probability = probability
    def __call__(self, data):
        if np.random.random() <= self.probability:
            return np.flip(data,axis=0).copy()
        return data


class InversePosesPre(object):
    '''
    Left-right flip of skeletons
    '''
    def __init__(self, probability=0.5, joint_format='coco'):
        self.probability = probability
        if joint_format == 'coco':
            self.invers_arr = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        elif joint_format in ['alphapose', 'openpose']:
            self.invers_arr = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]

    def __call__(self, data):
        for i in range(len(data)):
            if np.random.random() <= self.probability:
                data[i]=data[i,self.invers_arr,:]
        return data


class JointNoise(object):
    """
    Add Gaussian noise to joint
    std: standard deviation
    """

    def __init__(self, std=0.25):
        self.std = std

    def __call__(self, data):
        # T, V, C
        noise = np.hstack((
            np.random.normal(0, self.std, (data.shape[1], 2)),
            np.zeros((data.shape[1], 1))
        )).astype(np.float32)

        return data + np.repeat(noise[np.newaxis, ...], data.shape[0], axis=0)


class GaitTR_MultiInput(object):
    def __init__(self, joint_format='coco',):
        if joint_format == 'coco':
            self.connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
        elif joint_format in ['alphapose', 'openpose']:
            self.connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])


    def __call__(self, data):
        # (C, T, V) -> (I, C * 2, T, V)
        data = np.transpose(data, (2, 0, 1))

        data = data[:2, :, :]

        C, T, V = data.shape
        data_new = np.zeros((5, C, T, V))
        # Joints
        data_new[0, :C, :, :] = data
        for i in range(V):
            data_new[1, :, :, i] = data[:, :, i] - data[:, :, 0]
        # Velocity
        for i in range(T - 2):
            data_new[2, :, i, :] = data[:, i + 1, :] - data[:, i, :]
            data_new[3, :, i, :] = data[:, i + 2, :] - data[:, i, :]
        # Bones
        for i in range(len(self.connect_joint)):
            data_new[4, :, :, i] = data[:, :, i] - data[:, :, self.connect_joint[i]]
        
        I, C, T, V = data_new.shape
        data_new = data_new.reshape(I*C, T, V)
        # (C T V) -> (T V C)
        data_new = np.transpose(data_new, (1, 2, 0))

        return data_new


class GaitGraph_MultiInput(object):
    def __init__(self, center=0, joint_format='coco'):
        self.center = center
        if joint_format == 'coco':
            self.connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
        elif joint_format in ['alphapose', 'openpose']:
            self.connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])

    def __call__(self, data):
        T, V, C = data.shape
        x_new = np.zeros((T, V, 3, C + 2))
        # Joints
        x = data
        x_new[:, :, 0, :C] = x
        for i in range(V):
            x_new[:, i, 0, C:] = x[:, i, :2] - x[:, self.center, :2]
        # Velocity
        for i in range(T - 2):
            x_new[i, :, 1, :2] = x[i + 1, :, :2] - x[i, :, :2]
            x_new[i, :, 1, 3:] = x[i + 2, :, :2] - x[i, :, :2]
        x_new[:, :, 1, 3] = x[:, :, 2]
        # Bones
        for i in range(V):
            x_new[:, i, 2, :2] = x[:, i, :2] - x[:, self.connect_joint[i], :2]
        # Angles
        bone_length = 0
        for i in range(C - 1):
            bone_length += np.power(x_new[:, :, 2, i], 2)
        bone_length = np.sqrt(bone_length) + 0.0001
        for i in range(C - 1):
            x_new[:, :, 2, C+i] = np.arccos(x_new[:, :, 2, i] / bone_length)
        x_new[:, :, 2, 3] = x[:, :, 2]
        
        return x_new


class SkeletonInput(object):
    '''
    Transpose the input
    '''
    def __call__(self, data):
        # (T V C) -> (C T V)
        data = np.transpose(data, (2, 0, 1))
        return data[...,np.newaxis]


def get_transform(trf_cfg=None):
    if is_dict(trf_cfg):
        transform = getattr(base_transform, trf_cfg['type'])
        valid_trf_arg = get_valid_args(transform, trf_cfg, ['type'])
        return transform(**valid_trf_arg)
    if trf_cfg is None:
        return lambda x: x
    if is_list(trf_cfg):
        transform = [get_transform(cfg) for cfg in trf_cfg]
        return transform
    raise "Error type for -Transform-Cfg-"


class TwoView(object):
    def __init__(self,trf_cfg):
        assert is_list(trf_cfg)
        self.transform = T.Compose([get_transform(cfg) for cfg in trf_cfg])
    def __call__(self, data):
        return np.concatenate([self.transform(data), self.transform(data)], axis=1)


def Compose(trf_cfg):
    assert is_list(trf_cfg)
    transform = T.Compose([get_transform(cfg) for cfg in trf_cfg])
    return transform