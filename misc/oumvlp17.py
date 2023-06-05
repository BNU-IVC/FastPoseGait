import pickle
from tqdm import tqdm
import os
import os.path as osp
import argparse

'''
    gernerate the COCO Pose Format from 18 Number of Points
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
    parser = argparse.ArgumentParser(description="Preprocessing for OUMVLP Pose from 18 point format to 17 COCO format")
    #add
    parser.add_argument("--datasetdir",help="The dir of the pose dataset")
    parser.add_argument("--dstdir",help="Target dir")
    #groups
    opt = parser.parse_args()
    #return 
    return opt

def ToCOCO(dataset_dir,dstdir):
    '''
    COCO
               # keypoints = {
            #     0: "nose",
            #     1: "left_eye",
            #     2: "right_eye",
            #     3: "left_ear",
            #     4: "right_ear",
            #     5: "left_shoulder",
            #     6: "right_shoulder",
            #     7: "left_elbow",
            #     8: "right_elbow",
            #     9: "left_wrist",
            #     10: "right_wrist",
            #     11: "left_hip",
            #     12: "right_hip",
            #     13: "left_knee",
            #     14: "right_knee",
            #     15: "left_ankle",
            #     16: "right_ankle"
            # }
    OUMVLP
    mask=[0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10]
           # keypoints = {
            #     0: "nose",  
            #     1: "neck"
            #     2: "Rshoulder"
            #     3: "Relbow"
            #     4: "Rwrist"
            #     5: "Lshoudler"
            #     6: "Lelbow"
            #     7: "Lwrist"
            #     8: "Rhip
            #     9: "Rknee"
            #     10: "Rankle"
            #     11: "Lhip"
            #     12: "Lknee"
            #     13: "Lankle"
            #     14: "Reye"
            #     15: "Leye"
            #     16: "Rear"
            #     17: "Lear"
            # }
    '''
    mask=[0,15,14,17,16,5,2,6,3,7,4,11,8,12,9,13,10]
    id_list = sorted(os.listdir(dataset_dir))
    for id_ in tqdm(id_list):
        path_id = osp.join(dataset_dir,id_)
        ViewSeqList = sorted(os.listdir(path_id))
        for ViewSeq_ in ViewSeqList:
            path_ViewSeq = osp.join(path_id,ViewSeq_)
            Sequences = sorted(os.listdir(path_ViewSeq))
            #frame_list: temporally stored the seqs
            #frame_list = []
            for seq in Sequences:
                seq_path = osp.join(path_ViewSeq,seq,f'{seq}.pkl')
                with open(seq_path,'rb') as f:
                    data = pickle.load(f)
                    #data [T,18,3]
                new_data = data[...,mask,:].copy()
                #[T,17,3]
                subject = id_
                sequence = ViewSeq_
                view = seq
                #dst_root =dstdir
                dst_path = os.path.join(dstdir,subject,sequence,view)
                os.makedirs(dst_path,exist_ok=True)
                pkl_path = os.path.join(dst_path,f'{view}.pkl')
                pickle.dump(new_data,open(pkl_path,'wb')) 
    return True
'''
The Main Function
'''
if __name__ == '__main__':
    #get the option
    opt = parse_option()

    if ToCOCO(opt.datasetdir,opt.dstdir):
        datasetName = opt.datasetdir.split('/')[-1]
        print(f'Success! The dataset: {datasetName} has been Processed')
    else:
        print('Failed to Convert!')