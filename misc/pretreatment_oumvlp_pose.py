import os
import json
import os.path as osp
from tqdm import tqdm
import argparse
import numpy as np
import pickle
'''
How to use it?
    Script Example:
    #python pretreamt_oumvlp_pose.py --datasetdir=<YOUR OUMVLP DATASET PATH> --dstdir=<YOUR TARGET PATH>
This code is used for Preprocess the OUMVLP Pose dataset json file into
PKL file

        Original Dataset
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
        Processed Dataset
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
    #groups
    opt = parser.parse_args()
    #return
    return opt

def Json2PKL(dataset_dir,dstdir):
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
                if len(data['people'])==0:
                    continue
                pose = np.array(data["people"][0]["pose_keypoints_2d"]).reshape(-1,3)
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
    if Json2PKL(opt.datasetdir,opt.dstdir):
        datasetName = opt.datasetdir.split('/')[-1]
        print(f'Success! The dataset: {datasetName} has been Processed')
    else:
        print('Failed to Convert!')
    

    