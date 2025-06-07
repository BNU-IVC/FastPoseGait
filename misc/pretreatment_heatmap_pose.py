import mmcv
from mmcv import imread
import mmengine
from mmengine.registry import init_default_scope
import numpy as np

import os
import pickle
import os.path as osp
from tqdm import tqdm

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples


try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

local_runtime = False

try:
    from google.colab.patches import cv2_imshow  # for image visualization in colab
except:
    local_runtime = True


pose_config = 'configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py'
pose_checkpoint = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth'
det_config = 'projects/rtmpose/rtmdet/person/rtmdet_nano_320-8xb32_coco-person.py'
det_checkpoint = 'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth'

device = 'cuda:0'
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))


# build detector
detector = init_detector(
    det_config,
    det_checkpoint,
    device=device
)


# build pose estimator
pose_estimator = init_pose_estimator(
    pose_config,
    pose_checkpoint,
    device=device,
    cfg_options=cfg_options
)

# init visualizer
pose_estimator.cfg.visualizer.radius = 3
pose_estimator.cfg.visualizer.alpha = 0.8
pose_estimator.cfg.visualizer.line_width = 1

visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
visualizer.set_dataset_meta(
    pose_estimator.dataset_meta, skeleton_style='mmpose')

# main fucntion
def process(img_path):
    scope = detector.cfg.get('default_scope', 'mmdet')
    if scope is not None:
        init_default_scope(scope)
    detect_result = inference_detector(detector, img_path)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                    pred_instance.scores > 0.3)]
    bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)

    # # show the results
    # img = imread(img_path, channel_order='rgb')
    # visualizer.add_datasample(
    #     'result',
    #     img,
    #     data_sample=data_samples,
    #     draw_gt=False,
    #     draw_bbox=True,
    #     kpt_thr=0.3,
    #     draw_heatmap=True,
    #     show_kpt_idx=True,
    #     skeleton_style='mmpose',
    #     show=False,
    #     out_file='vis.jpg')

    keypoints = pose_results[0].pred_instances.keypoints #[1,17,2]
    keypoints_score = pose_results[0].pred_instances.keypoint_scores#[1,17]
    
    heatmaps = pose_results[0]._pred_heatmaps.heatmaps.cpu().numpy()#[17,64,48]


    #heatmaps_float
    heatmaps_float = heatmaps.astype(np.float16)

    return np.concatenate((keypoints[0],keypoints_score[0,:,None]),axis=-1), heatmaps_float


dataset_dir = 'RGB_DATASET_PATH'
dstdir = 'POSE_HEATMAP_PATH'
id_list = sorted(os.listdir(dataset_dir),reverse=False)

# You can use multiple GPUs here for processing.
start = int(len(id_list)/8 * 0)
end = int(len(id_list)/8* 8)


for id_ in tqdm(id_list[start:end]):

    path_id = osp.join(dataset_dir,id_)
    ViewSeqList = sorted(os.listdir(path_id))
    for ViewSeq_ in ViewSeqList:
        path_ViewSeq = osp.join(path_id,ViewSeq_)
        Sequences = sorted(os.listdir(path_ViewSeq))
        for seq in Sequences:
            path_imgs = osp.join(path_ViewSeq,seq)
            img_seqs = sorted(os.listdir(path_imgs))
            keypoints_list = []
            float_list = []

            if os.path.exists(osp.join(dstdir,id_,ViewSeq_,seq)):
                print(f'the file {path_imgs} exits')
                continue
            for img in img_seqs:
                img_path = osp.join(path_imgs,img)

                keypoints,heatmap_float = process(img_path)
                keypoints_list.append(keypoints)
                float_list.append(heatmap_float)

            
            keypoints_re = np.stack(keypoints_list,axis=0)
            float_re = np.stack(float_list, axis=0)
            
            subject = id_
            sequence = ViewSeq_
            view = seq
            dst_path = os.path.join(dstdir,subject,sequence,view)
            os.makedirs(dst_path,exist_ok=True)
            
            pkl_path = os.path.join(dst_path,f'pose_{view}.pkl')
            pickle.dump(keypoints_re,open(pkl_path,'wb'))
            print(pkl_path) 
            
            pkl_path = os.path.join(dst_path,f'float_{view}.pkl')
            pickle.dump(float_re,open(pkl_path,'wb'))
            print(pkl_path) 
            
