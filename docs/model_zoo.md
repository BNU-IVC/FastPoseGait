# Model Zoo

##  [CASIA-B](https://ieeexplore.ieee.org/abstract/document/1699873/)
|                       Model            |  Pose Estimator             |  Rank-1 NM  |  Rank-1 BG  |  Rank-1 CL  | Rank-1 Mean      |
| :------------------------------------------------: | :---------: | :---------: | :---------: | :---------: | ------------ |
|                   [GaitGraph1](https://ieeexplore.ieee.org/document/9506717)              | HRNet     | 86.37 (87.7) | 76.50 (74.8) | 65.24 (66.3) | 76.04 (76.27) |
| [GaitGraph2](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper)  |HRNet | 80.58 (82.0) | 72.14 (73.2) | 65.26 (63.6) | 72.66 (72.93) |
|                     [GaitTR]()   |        SimCC           |      94.91 (96.0) | 88.82 (91.3) | 90.34 (90.0) | 91.35 (92.4)          |



##  [OUMVLP-Pose](https://ieeexplore.ieee.org/abstract/document/9139355/)

|                       Model                      |  Pose Estimator   |  Rank-1 (original format) | Rank-1 (COCO2017 format)  
| :------------------------------------------------: | :------: | :------: | :------: 
|                   [GaitGraph1](https://ieeexplore.ieee.org/document/9506717)                   | AlphaPose| 2.81         | 4.24
|                   [GaitGraph2](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper)                  |AlphaPose/OpenPose| 62.11/49.02 |70.68/55.02
|                     [GaitTR](https://arxiv.org/abs/2204.03873)                     |AlphaPose|   40.30    |39.77





##  [GREW](http://openaccess.thecvf.com/content/ICCV2021/html/Zhu_Gait_Recognition_in_the_Wild_A_Benchmark_ICCV_2021_paper.html)

|                       Model                       | Pose Estimator  | Rank-1  |
| :------------------------------------------------:| :-----: | :-----: |
|                   [GaitGraph1](https://ieeexplore.ieee.org/document/9506717)                  |HRNet |     10.18    |
|                   [GaitGraph2](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper)                  |HRNet |   34.77    |
|                     [GaitTR](https://arxiv.org/abs/2204.03873)                    |HRNet |   48.58   |



##  [Gait3D](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Gait_Recognition_in_the_Wild_With_Dense_3D_Representations_and_CVPR_2022_paper.html)


|                       Model                       |Pose Estimator | Rank-1  |
| :------------------------------------------------:| :-----: | :-----: |
|                   [GaitGraph1](https://ieeexplore.ieee.org/document/9506717)                  | HRNet |   8.60    |
|                   [GaitGraph2](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper)                 | HRNet |   11.20      |
|                     [GaitTR](https://arxiv.org/abs/2204.03873)                  | HRNet  |   7.20    |




The results in the parentheses are mentioned in the papers. 

**Note:**
* We sincerely thank all authors for their excellent works.
* We modified the training strategy of GaitGraph1/GaitGraph2 and did not select the best model during training for the initialization of the second phase. Please refer to [configs](../configs) for the details.

