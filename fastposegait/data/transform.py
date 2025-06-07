from data import transform as base_transform
import numpy as np
import math
from math import atan2, degrees, radians, cos, sin
from utils import is_list, is_dict, get_valid_args
import torchvision.transforms as T

############################  ---START--- add by fuyang  ---START---  ############################

# to balance the distrubution for each image
####  Mean Data Normalization  ####
class Sequence_level():
    def __call__(self, data):

        max_v = np.max(data) # [1]
        min_v = np.min(data) #[1]
        # 进行均值归一化
        normalized_data = (data - min_v) / (max_v - min_v) #[T, H, W]
        
        return normalized_data

# to generate the rotation matrix
#### angle: spine (composed by neck and hip)
#### center: neck

class RotationMat():
    def __call__(self, data):
        w = 48
        h = 64
        rotation_matrices = []
        for i in range(len(data)):            
            down_p = (data[i,11] + data[i,12])/2
            up_p =  (data[i,5] + data[i,6])/2
            dx, dy = down_p[0] - up_p[0], down_p[1] - up_p[1]
            theta = degrees(atan2(dy, dx))
            angle= -(90-theta)
            if np.abs(angle)<=5 and random.uniform(0, 1)>0.2:
                M_cv = np.array([[1, 0, 0],
                        [0, 1, 0]],dtype=np.float32)
                M_b = np.array([[1, 0, 0],
                        [0, 1, 0]],dtype=np.float32)
                M = np.stack([M_cv, M_b],axis=-1) 
                rotation_matrices.append(M)
                continue #next frame
            center = (down_p+up_p)/2
            M_cv = cv2.getRotationMatrix2D(center[:2], angle, 1.0)
            MM = np.concatenate([M_cv,np.array([[0,0,1]])],axis=0)
            #transform pose
            new_up_p = affine_transform(M_cv,up_p[:2])
            T = np.array([[2 / w, 0, -1],
                        [0, 2 / h, -1],
                        [0, 0, 1]])
            M_cv = np.linalg.inv(T @ MM @ np.linalg.inv(T))[:2]

            x_bias = -(new_up_p[0] - 24)
            y_bias = -(new_up_p[1] - 16)
            M_b = np.array([[1, 0, x_bias],
                            [0, 1, y_bias],
                            [0,0,1]])
            M_b = np.linalg.inv(T @ M_b @ np.linalg.inv(T))[:2]   

            M = np.stack([M_cv, M_b],axis=-1) # 2,3,2
            rotation_matrices.append(M)
        
        return np.array(rotation_matrices)



# to generate the rotation matrix and alignment
##==> First
#### angle: spine (composed by neck and hip)
#### center: neck
##==> Second
#### apply the rotation mat to pose
#### new bias to neck point
def affine_transform(matrix, point):
    """
    Apply a 2x3 affine transformation matrix to a 2D point.

    Parameters:
    matrix (np.array): A 2x3 affine transformation matrix.
    point (tuple): A tuple (x, y) representing the point to be transformed.

    Returns:
    np.array: The transformed point as a numpy array.
    """
    point = np.array([point[0], point[1], 1])  # Convert to a 3-element array
    affine_matrix = np.vstack([matrix, [0, 0, 1]])  # Convert to a 3x3 matrix
    transformed_point = np.dot(affine_matrix, point)  # Apply transformation
    return transformed_point[:2]

def create_offset_matrix(transformed_point, target_point):
    """
    Create a 2x3 affine transformation matrix for the offset from the transformed point to the target point.

    Parameters:
    transformed_point (np.array): The transformed point.
    target_point (tuple): The target point to calculate the offset to.

    Returns:
    np.array: A 2x3 affine transformation matrix for the calculated offset.
    """
    offset = np.array(target_point) - transformed_point
    offset_matrix = np.array([
        [1, 0, offset[0]],
        [0, 1, offset[1]]
    ])
    return offset_matrix

class AffineMat():
    def __call__(self, data):
        rotation_matrices = []
        for i in range(len(data)):
            down_p = (data[i,11] + data[i,12])/2 # hip 
            up_p =  (data[i,5] + data[i,6])/2 # neck
            dx, dy = down_p[0] - up_p[0], down_p[1] - up_p[1] #spine
            theta = degrees(atan2(dy, dx)) # angle
            angle= (theta -90) #rotated angle
            angle = angle/180*math.pi # theta
            rotation_M = np.array([[math.cos(angle),math.sin(-angle),0],
                        [math.sin(angle),math.cos(angle) ,0]]) 
            ##apply the M to neck
            ## fixed neck point (24,24)
            ## bias neck - (24,24)
            
            new_up_p = affine_transform(rotation_M,up_p[:2])
            new_up_p[0] 
            offset_M = create_offset_matrix(new_up_p,(24,24))
            M = np.stack([rotation_M, offset_M],axis=-1) # 2,3,2
            rotation_matrices.append(M)
        
        return np.array(rotation_matrices) # T 2,3,2

############################  ----END --- add by fuyang ----END ---  ############################


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
    def __init__(self, joint_format='coco'):

        if joint_format == 'coco':
            self.connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
            self.center = 0
        elif joint_format in ['alphapose', 'openpose']:
            self.connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            self.center = 1

    def __call__(self, data):
        # (C, T, V) -> (I, C * 2, T, V)
        data = np.transpose(data, (2, 0, 1))

        data = data[:2, :, :]

        C, T, V = data.shape
        data_new = np.zeros((5, C, T, V))
        # Joints
        data_new[0, :C, :, :] = data
        for i in range(V):
            data_new[1, :, :, i] = data[:, :, i] - data[:, :, self.center]
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
    def __init__(self, joint_format='coco'):
        
        if joint_format == 'coco':
            self.connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
            self.center = 0
        elif joint_format in ['alphapose', 'openpose']:
            self.connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            self.center = 1

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


class HOD_MultiInput(object):
    '''
    Human-oriented descripter of GPGait
    Paper: https://arxiv.org/abs/2303.05234
    '''
    def __init__(self, joint_format='coco'):
        if joint_format == 'coco':
            self.connect_joint = np.array([5,0,0,1,2,0,0,5,6,7,8,5,6,11,12,13,14])
            self.angle_list = [
                np.array([0,1,2]),
                np.array([1,0,3]),
                np.array([2,4,0]),
                np.array([3,1]),
                np.array([4,2]),
                np.array([5,7,11]),
                np.array([6,8,12]),
                np.array([7,5,9]),
                np.array([8,10,6]),
                np.array([9,7]),
                np.array([10,8]),
                np.array([11,5,13]),
                np.array([12,6,14]),
                np.array([13,11,15]),
                np.array([14,12,16]),
                np.array([15,13]),
                np.array([16,14])
            ]
        elif joint_format == 'alphapose' or joint_format == 'openpose':
            self.connect_joint = np.array([1,1,1,2,3,1,5,6,2,8,9,5,11,12,0,0,14,15])
            self.angle_list = [
                np.array([0,15,14]),
                np.array([15,0,17]),
                np.array([14,16,0]),
                np.array([17,15]),
                np.array([16,14]),
                np.array([5,6,11]),
                np.array([2,3,8]),
                np.array([6,3,8]),
                np.array([3,4,2]),
                np.array([7,6]),
                np.array([4,3]),
                np.array([11,5,12]),
                np.array([8,2,9]),
                np.array([12,11,13]),
                np.array([9,8,10]),
                np.array([13,12]),
                np.array([10,9])
            ]
    
    def cal_edge_angle(self,center,left):
        # generate center/left/right
        right = np.zeros_like(center)
        right[0,:] = center[0,:]
        right[1,:] = left[1,:]
        return self.cos_law(center,left,right)

    def cal_inner_angle(self,center,left,right):
        return self.cos_law(center,left,right)

    def cos_law(self,center,left,right):
        # adjacent_side_1
        side1 = np.sqrt(np.power(center[0,:]-left[0,:],2) + np.power(center[1,:]-left[1,:],2))
        # adjacent_side_2
        side2 = np.sqrt(np.power(center[0,:]-right[0,:],2) + np.power(center[1,:]-right[1,:],2))
        # opposite_side
        side3 = np.sqrt(np.power(left[0,:]-right[0,:],2) + np.power(left[1,:]-right[1,:],2))
        # cosine_law
        deno = side1*side2
        where_zero = np.where(deno==0)
        deno[where_zero] = 1
        cos = (side1*side1+side2*side2-side3*side3)/(2*deno)
        where_mo_one = np.where(cos > 1)
        where_le_one = np.where(cos < -1)
        cos[where_mo_one] = 1
        cos[where_le_one] = -1
        data_return = math.pi - np.arccos(cos)
        data_return[where_zero] = math.pi
        return data_return
    
    def __call__(self, data):
        data = np.transpose(data, (2, 0, 1))
        data = data[:2, :, :]

        C, T, V = data.shape
        data_new = np.zeros((3, C, T, V))
        # Joints
        data_new[0, :C, :, :] = data
        # Bones
        for i in range(len(self.connect_joint)):
            data_new[1, :, :, i] = data[:, :, i] - data[:, :, self.connect_joint[i]]
        # Angles
        for i in range(len(self.angle_list)):
            if len(self.angle_list[i]) == 3:
                data_new[2,:1,:,i] = self.cal_inner_angle(data[:,:,self.angle_list[i][0]],data[:,:,self.angle_list[i][1]],data[:,:,self.angle_list[i][2]])
            else:
                data_new[2,:1,:,i] = self.cal_edge_angle(data[:,:,self.angle_list[i][0]],data[:,:,self.angle_list[i][1]])
        
        data_new = np.nan_to_num(data_new)

        I, C, T, V = data_new.shape
        data_new = data_new.reshape(I*C, T, V)
        # (C T V) -> (T V C)
        data_new = np.transpose(data_new, (1, 2, 0))
        
        return data_new




class Affine(object):
    '''
    Make the spine of skeleton vertical to the ground
    Detailed description in paper: https://arxiv.org/abs/2303.05234

    The formula for counterclockwise rotation around a point (x0, y0):
    xx = x * cos - y * sin + x0 * (1-cos) + y0 * sin
    yy = x * sin + y * cos + y0 * (1-cos) - x0 * sin
    '''

    def __init__(self,fi=0):
        self.fi = fi

    def __call__(self, data):
        kp = data
        kp_x = kp[..., 0]
        kp_y = kp[..., 1]
        neck_x = (kp_x[:, 5]+kp_x[:, 6]) / 2
        neck_y = (kp_y[:, 5]+kp_y[:, 6]) / 2
        hip_x = (kp_x[:, 11] + kp_x[:, 12]) / 2
        hip_y = (kp_y[:, 11] + kp_y[:, 12]) / 2
        
        x_l = hip_x - neck_x
        y_l = hip_y - neck_y 
        where_zero = np.where(y_l==0)
        y_l[where_zero] = 1
        theta = np.arctan(x_l / y_l)
        theta[where_zero] = 0.
        theta = np.expand_dims(theta, axis=-1)
        neck_x = np.expand_dims(neck_x, axis=-1)
        neck_y = np.expand_dims(neck_y, axis=-1)
        # if theta > self.fi:
        kp[..., 0] = np.cos(theta) * kp[..., 0] - np.sin(theta) * kp[..., 1] + (1 - np.cos(theta)) * neck_x + np.sin(theta) * neck_y
        kp[..., 1] = np.sin(theta) * kp[..., 0] + np.cos(theta) * kp[..., 1] + (1 - np.cos(theta)) * neck_y - np.sin(theta) * neck_x
        data = kp
        return data


class RescaleCenter(object):
    '''
    Normalize and align the skeletons
    Detailed description in paper: https://arxiv.org/abs/2303.05234
    '''
    def __init__(self,center='neck',scale=225):
        self.center = center
        self.scale = scale

    def __call__(self, data):
        
        kp = data # [T,17,3]

        if self.center=='None' or self.center ==None:
            return kp
        
        # Rescale
        kp_x = kp[...,0]
        kp_y = kp[...,1]
        min_y = np.min(kp_y, axis=1).reshape(-1, 1)
        max_y = np.max(kp_y, axis=1).reshape(-1, 1)
        old_h = max_y - min_y
        new_h = self.scale
        projection =  new_h / old_h
        kp_x *= projection
        kp_y *= projection

        # Center
        if self.center=='neck':
            offset_x = (kp_x[:, 5]+kp_x[:, 6])/2
            offset_y = (kp_y[:, 5]+kp_y[:, 6])/2
        elif self.center =='head':
            offset_x = (kp_x[:, 1]+kp_x[:, 2])/2
            offset_y = (kp_y[:, 1]+kp_y[:, 2])/2
        elif self.center =='hip':
            offset_x = (kp_x[:, 11]+kp_x[:, 12])/2
            offset_y = (kp_y[:, 11]+kp_y[:, 12])/2
        #offset
        kp_x -= offset_x.reshape(-1, 1)
        kp_y -= offset_y.reshape(-1, 1)
        data = kp
        return data


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