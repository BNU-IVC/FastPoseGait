import os
import os.path as osp
from tqdm import tqdm
import argparse
import numpy as np
import pickle
'''
How to use it?
    Script Example:
    python pretreatment_grew_pose.py --datasetdir=/Dataset/GREW/train --dstdir=/Dataset/GREW-pkl
    python pretreatment_grew_pose.py --datasetdir=/Dataset/GREW/test/gallery --dstdir=/Dataset/GREW-pkl
    python pretreatment_grew_pose.py --datasetdir=/Dataset/GREW/test/probe --dstdir=/Dataset/GREW-pkl
This code is used for Preprocess the Gait3D Pose dataset TXT file into PKL file

        Original Dataset
            ├── make_pose(Silhouettes, Gait Energy Images (GEIs) , 2D and 3D poses)
                ├── train
                    ├── 00001
                        ├── 4XPn5Z28
                            ├── 00001.png
                            ├── 00001_2d_pose.txt
                            ├── 00001_3d_pose.txt
                        ├── 4XPn5Z28_gei.png
                ├── test
                    ├── gallery
                        ├── 00001
                            ├── 79XJefi8
                                ├── 00001.png
                                ├── 00001_2d_pose.txt
                                ├── 00001_3d_pose.txt
                            ├── 79XJefi8_gei.png
                    ├── probe
                        ├── 01DdvEHX
                            ├── 00001.png
                            ├── 00001_2d_pose.txt
                            ├── 00001_3d_pose.txt
                        ├── 01DdvEHX_gei.png
        Processed Dataset
            GREW-pkl
                ├── 00001train (subject in training set)
                    ├── 00
                        ├── 4XPn5Z28
                            ├── 4XPn5Z28.pkl
                        ├──5TXe8svE
                            ├── 5TXe8svE.pkl
                            ......
                ├── 00001 (subject in testing set)
                    ├── 01
                        ├── 79XJefi8
                            ├── 79XJefi8.pkl
                    ├── 02
                        ├── t16VLaQf
                            ├── t16VLaQf.pkl
                ├── probe
                    ├── 03
                        ├── etaGVnWf
                            ├── etaGVnWf.pkl
                        ├── eT1EXpgZ
                            ├── eT1EXpgZ.pkl   
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
    parser = argparse.ArgumentParser(description="Preprocessing for GREW Pose")
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
    if dataset_dir.split("/")[-1] == 'probe':
        id_list = sorted(os.listdir(dataset_dir))
        for id_ in tqdm(id_list):
            path_id = osp.join(dataset_dir,id_)
            ViewSeqList = sorted(os.listdir(path_id))
            frame_list = []
            for seq in ViewSeqList:
                if len(seq.split('_')) > 1:
                    type_name = seq.split('_')[1]
                    if type_name == '2d':
                        frame_path = osp.join(path_id, seq)
                        data = np.genfromtxt(frame_path, delimiter=',')[2:].reshape(-1,3)
                        frame_list.append(data)

            if not frame_list:
                print(f'Invaild sequence:{path_id}')
                continue
            keypoints = np.stack(frame_list)
            index = '03'
            dst_path = osp.join(dstdir, index, id_)
            os.makedirs(dst_path, exist_ok=True)
            pkl_path = os.path.join(dst_path,f'{id_}.pkl')
            pickle.dump(keypoints, open(pkl_path,'wb'))
    else:
        id_list = sorted(os.listdir(dataset_dir))
        for id_ in tqdm(id_list):
            path_id = osp.join(dataset_dir,id_)
            ViewSeqList = sorted(os.listdir(path_id))
            index_int = 0
            for ViewSeq_ in ViewSeqList:
                if ViewSeq_.split('.')[-1] != 'png':
                    path_ViewSeq = osp.join(path_id,ViewSeq_)
                else:
                    continue
                Sequences = sorted(os.listdir(path_ViewSeq))
                frame_list = []
                for seq in Sequences:
                    if len(seq.split('_')) > 1:
                        type_name = seq.split('_')[1]
                        if type_name == '2d':
                            frame_path = osp.join(path_ViewSeq, seq)
                            data = np.genfromtxt(frame_path, delimiter=',')[2:].reshape(-1,3)
                            frame_list.append(data)
                    # stack frames
                    # get keypoint of shape[T,17,3]
                    # NULL -> continue
                if not frame_list:
                    print(f'Invaild sequence:{path_ViewSeq}')
                    continue
                keypoints = np.stack(frame_list)
                ##dst path elements
                if dataset_dir.split("/")[-1] == 'train':
                    id_type = id_ + 'train'
                    index = '00'
                elif dataset_dir.split("/")[-1] == 'gallery':
                    id_type = id_
                    index_int = index_int + 1
                    index = str(index_int).zfill(2)
                dst_path = osp.join(dstdir, id_type, index, ViewSeq_)
                os.makedirs(dst_path, exist_ok=True)
                pkl_path = os.path.join(dst_path,f'{ViewSeq_}.pkl')
                pickle.dump(keypoints, open(pkl_path,'wb'))
    
    return True

'''
The Main Function
'''
if __name__ == '__main__':
    #get the option
    opt = parse_option()

    if TXT2PKL(opt.datasetdir, opt.dstdir):
        datasetName = opt.datasetdir.split('/')[-1]
        print(f'Success! The dataset: {datasetName} has been Processed')
    else:
        print('Failed to Convert!')
    

    