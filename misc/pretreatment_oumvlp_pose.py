import os
import re
import cv2
import json
import os.path as osp
from tqdm import tqdm
import argparse
import numpy as np
import pickle
'''
How to use it?
    Script Example:
        python pretreamt_oumvlp_pose.py \
        --datasetdir=<YOUR OUMVLP DATASET PATH> \
        --dstdir=<YOUR TARGET PATH> \
        --siluset_dir=<YOUR OUMVLP SILU DATASET PATH>
This code is used for Preprocess the OUMVLP Pose dataset json file into
PKL file
        Original Silu Dataset
            input data format
                Silhouette_000-00 (view-sequence)
                    00001 (subject) 
                        0001.png
                        0002.png
                        ...
                    00002 (subject)
                        0001.png
                        0002.png
                        ...
                    ...
                Silhouette_000-01 (view-sequence)
                    ...
            
        Original Pose Dataset
            input data format
                alphapose(dataset_root)
                00001(subject)
                    000_00(view-sequence)
                        0021_keypoint.json
                        0022_keypoint.json
                        ...
                    000_01(view-sequence)
                    ...
                00002(subject)
                    ...

        Processed Pose Dataset
            outout data format
            alhpapose(dataset_root)
                00001(subject)
                    00(sequence)
                        000(view)
                            000.pkl(contains all frames)
                        015(view)
                            015.plk(contains all frames)
                    01(sequence)
                00002(subject)       
'''

def parse_option():
    '''
    Get the dataset path from command
        input args:
            -dataset path
        return:
            opt
 
    '''
    #init
    parser = argparse.ArgumentParser(description="Preprocessing for OUMVLP Pose")
    #add
    parser.add_argument("--datasetdir",help="The dir of the pose dataset")
    parser.add_argument("--dstdir",help="Target dir")
    parser.add_argument("--siluset_dir",type=str, default="",
                        help="The dir of the origin oumvlp silu dataset")
    #groups
    opt = parser.parse_args()
    
    opt.datasetdir = osp.abspath(opt.datasetdir)
    opt.siluset_dir = osp.abspath(opt.siluset_dir)
    assert osp.exists(opt.datasetdir), f"The pose dataset({opt.datasetdir}) is not exists"
    assert osp.exists(opt.siluset_dir), f"The silu dataset({opt.siluset_dir}) is not exists"
    #return
    return opt

def pose_silu_match_score(pose: np.ndarray, silu: np.ndarray):
    """Compute the sum of the pixel intensity on all joints in the silu image as the silu-pose matching score.

    Args:
        pose (np.ndarray): the pose data with shape (17,3)
        silu (np.ndarray): the origin silu frame with shape (H,W) or (H,W,C)

    Returns:
        float: the matching score of given pose and silu
    """
    pose_coord = pose[:,:2].astype(np.int32)
    
    H, W, *_ = silu.shape
    valid_joints = (pose_coord[:, 1] >=0) & (pose_coord[:, 1] < H) & \
                   (pose_coord[:, 0] >=0) & (pose_coord[:, 0] < W)
    if np.sum(valid_joints) == len(pose_coord):
        # only calculate score for points that are inside the silu img
        # use the sum of all joints' pixel intensity as the score
        return np.sum(silu[pose_coord[:, 1], pose_coord[:, 0]])
    else:
        # if pose coord is out of bound, return -inf
        return -np.inf

def Json2PKL(dataset_dir,dstdir, siluset_dir):
    '''
    Convert the jsons into PKL
    See the Description above
    '''
    id_list = sorted(os.listdir(dataset_dir))
    for id_ in tqdm(id_list):
        path_id = osp.join(dataset_dir,id_)
        ViewSeqList = sorted(os.listdir(path_id))
        for ViewSeq_ in ViewSeqList:
            path_ViewSeq = osp.join(path_id,ViewSeq_)
            Sequences = sorted(os.listdir(path_ViewSeq))
            frame_list = []
            for seq in Sequences:
                seq_path = osp.join(path_ViewSeq,seq)
                with open(seq_path) as f:
                    data = json.load(f)
                    
                person_num = len(data['people'])
                if person_num==0:
                    continue
                elif person_num == 1:
                    pose = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1,3)
                else:
                    img_name = re.findall(r'\d{4}', osp.basename(seq_path))[-1] + '.png'
                    img_path = osp.join(siluset_dir, f"Silhouette_{ViewSeq_.replace('_', '-')}", id_, img_name)
                    if not os.path.exists(img_path):
                        print(f'Pose reference silu({img_path}) not exists.')
                        continue
                    silu_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    person_poses = [np.array(p["pose_keypoints_2d"]).reshape(-1,3) for p in data['people']]
                    max_score_idx = np.argmax([pose_silu_match_score(p, silu_img) for p in person_poses])
                    
                    pose = person_poses[max_score_idx]

                frame_list.append(pose)
            if not frame_list:
                null_name = path_ViewSeq.split('/')[-3:]
                print(f'Invaild sequence:{path_ViewSeq}')
                continue
            keypoints = np.stack(frame_list)
            subject = id_
            sequence = ViewSeq_.split('_')[1]
            view = ViewSeq_.split('_')[0]
            dst_root =dstdir
            dst_path = os.path.join(dstdir,subject,sequence,view)
            os.makedirs(dst_path,exist_ok=True)
            pkl_path = os.path.join(dst_path,f'{view}.pkl')
            pickle.dump(keypoints,open(pkl_path,'wb'))
            
    return True

'''
The Main Function
'''
if __name__ == '__main__':
    #get the option
    opt = parse_option()
    #use the Json2PKL function
    if Json2PKL(opt.datasetdir,opt.dstdir, opt.siluset_dir):
        datasetName = opt.datasetdir.split('/')[-1]
        print(f'Success! The dataset: {datasetName} has been Processed')
    else:
        print('Failed to Convert!')
    

    