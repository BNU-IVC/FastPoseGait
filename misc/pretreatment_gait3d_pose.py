import os
import os.path as osp
from tqdm import tqdm
import argparse
import numpy as np
import pickle
'''
How to use it?
    Script Example:
    python pretreatment_gait3d_pose.py --datasetdir=<YOUR Gait3D DATASET PATH> --dstdir=<YOUR TARGET DATASET PATH>
This code is used for Preprocess the Gait3D Pose dataset TXT file into PKL file

        Original Dataset
            input data format
            ├── 2D_Poses
            |  ├── 0000
            |      ├── camid0_videoid2
            |          ├── seq0
            |              ├── human_crop_f11627.txt
            |              ├── human_crop_f11628.txt
            |              ├── human_crop_f11629.txt
        Processed Dataset
            outout data format
            ├── 2D_Poses-pkl
            │  ├── 0000
            │     ├── camid0_videoid2
            │        ├── seq0
            │           └──seq0.pkl   
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
    parser = argparse.ArgumentParser(description="Preprocessing for Gait3D Pose")
    #add
    parser.add_argument("--datasetdir",help="The dir of the pose dataset")
    parser.add_argument("--dstdir",help="Target dir")
    #groups
    opt = parser.parse_args()
    #return
    return opt

def TXT2PKL(dataset_dir,dstdir):
    '''
    Convert the txt into PKL
    See the Description above
    '''
    id_list = sorted(os.listdir(dataset_dir))
    for id_ in tqdm(id_list):
        path_id = osp.join(dataset_dir,id_)
        ViewSeqList = sorted(os.listdir(path_id))
        for ViewSeq_ in ViewSeqList:
            path_ViewSeq = osp.join(path_id,ViewSeq_)
            Sequences = sorted(os.listdir(path_ViewSeq))
            for seq in Sequences:
                seq_path = osp.join(path_ViewSeq,seq)
                seq_list = sorted(os.listdir(seq_path))
                frame_list = []
                for frame in seq_list:
                    frame_path = osp.join(seq_path, frame)
                    data = np.genfromtxt(frame_path, delimiter=',')[2:].reshape(-1,3)
                    frame_list.append(data)
                # stack frames
                # get keypoint of shape[T,17,3]
                # NULL -> continue
                if not frame_list:
                    print(f'Invaild sequence:{seq_path}')
                    continue
                keypoints = np.stack(frame_list)
                ##dst path elements
                dst_path = osp.join(dstdir, id_, ViewSeq_, seq)
                os.makedirs(dst_path, exist_ok=True)
                pkl_path = os.path.join(dst_path,f'{seq}.pkl')
                pickle.dump(keypoints, open(pkl_path,'wb'))
            
    return True

'''
The Main Function
'''
if __name__ == '__main__':

    opt = parse_option()
    if TXT2PKL(opt.datasetdir, opt.dstdir):
        datasetName = opt.datasetdir.split('/')[-1]
        print(f'Success! The dataset: {datasetName} has been Processed')
    else:
        print('Failed to Convert!')
    

    