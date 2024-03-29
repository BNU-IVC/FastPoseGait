# Model Zoo

- [Vanilla Version](#vanilla-version)
- [Improved Version](#improved-version)

## Vanilla Version

###  [CASIA-B](https://ieeexplore.ieee.org/abstract/document/1699873/)
|                       Model            |  Pose Estimator             |  Rank-1 NM  |  Rank-1 BG  |  Rank-1 CL  | Rank-1 Mean      |
| :------------------------------------------------: | :---------: | :---------: | :---------: | :---------: | :------------: |
|                   [GaitGraph1](https://ieeexplore.ieee.org/document/9506717)              | HRNet     | 86.37 (87.7) | 76.50 (74.8) | 65.24 (66.3) | 76.04 (76.27) |
| [GaitGraph2](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper)  |HRNet | 80.29 (82.0) | 71.40 (73.2) | 63.80 (63.6) | 71.83 (72.93) |
|                     [GaitTR](https://arxiv.org/abs/2204.03873)   |        SimCC           |      94.91 (96.0) | 88.82 (91.3) | 90.34 (90.0) | 91.35 (92.4)          |
[GPGait](https://arxiv.org/abs/2303.05234)   |        HRNet           |      93.60 | 80.15 | 69.29 | 81.01          |




###  [OUMVLP-Pose](https://ieeexplore.ieee.org/abstract/document/9139355/)

|                       Model                      |  Pose Estimator   |  Rank-1 (original format) | Rank-1 (COCO2017 format)  
| :------------------------------------------------: | :------: | :------: | :------: 
|                   [GaitGraph1](https://ieeexplore.ieee.org/document/9506717)                   | AlphaPose| 2.81         | 4.24
|                   [GaitGraph2](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper)                  |AlphaPose/OpenPose| 62.11/49.02 |70.68/55.02
|                     [GaitTR](https://arxiv.org/abs/2204.03873)                     |AlphaPose|   40.30    |39.77
|                     [GPGait](https://arxiv.org/abs/2303.05234)                     |AlphaPose|   -   |59.11



###  [GREW](http://openaccess.thecvf.com/content/ICCV2021/html/Zhu_Gait_Recognition_in_the_Wild_A_Benchmark_ICCV_2021_paper.html)

|                       Model                       | Pose Estimator  | Rank-1  |
| :------------------------------------------------:| :-----: | :-----: |
|                   [GaitGraph1](https://ieeexplore.ieee.org/document/9506717)                  |HRNet |     10.18    |
|                   [GaitGraph2](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper)                  |HRNet |   34.78    |
|                     [GaitTR](https://arxiv.org/abs/2204.03873)                    |HRNet |   48.58   |
|                     [GPGait](https://arxiv.org/abs/2303.05234)                    |HRNet |   57.04   |



###  [Gait3D](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Gait_Recognition_in_the_Wild_With_Dense_3D_Representations_and_CVPR_2022_paper.html)


|                       Model                       |Pose Estimator | Rank-1  |
| :------------------------------------------------:| :-----: | :-----: |
|                   [GaitGraph1](https://ieeexplore.ieee.org/document/9506717)                  | HRNet |   8.60    |
|                   [GaitGraph2](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper)                 | HRNet |   11.20      |
|                     [GaitTR](https://arxiv.org/abs/2204.03873)                  | HRNet  |   7.20    |
|                     [GPGait](https://arxiv.org/abs/2303.05234)                  | HRNet  |   22.40   |



### [SUSTech1K](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_LidarGait_Benchmarking_3D_Gait_Recognition_With_Point_Clouds_CVPR_2023_paper.pdf)

| Model  | Pose estimator | Normal |  Bag  | Clothing | Carrying | Umbrella | Uniform | Occlusion | Night | Overall |
| :----: | :------------: | :----: | :---: | :------: | :------: | :------: | :-----: | :-------: | :---: | :-----: |
| [GPGait](https://arxiv.org/abs/2303.05234) |    ViTPose     | 49.91  | 46.91 |  33.06   |  45.79   |  40.82   |  51.7   |   66.57   | 30.81 |  47.38  |



### [CCPG](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_An_In-Depth_Exploration_of_Person_Re-Identification_and_Gait_Recognition_in_CVPR_2023_paper.pdf)

|                                            |                | CL-Full |       | CL-UP  |       | CL-DN  |       |
| :----------------------------------------: | :------------: | :-----: | :---: | :----: | :---: | :----: | :---: |
|                   Model                    | Pose estimator | Rank-1  |  mAP  | Rank-1 |  mAP  | Rank-1 |  mAP  |
| [GPGait](https://arxiv.org/abs/2303.05234) |     HRNet      |  54.75  | 25.78 | 65.60  | 38.44 | 71.06  | 41.04 |



## Improved Version

###  [CASIA-B](https://ieeexplore.ieee.org/abstract/document/1699873/)
|                       Model            |  Pose Estimator             |  Rank-1 NM  |  Rank-1 BG  |  Rank-1 CL  | Rank-1 Mean      |
| :------------------------------------------------: | :---------: | :---------: | :---------: | :---------: | ------------ |
|                   [GaitGraph1](https://ieeexplore.ieee.org/document/9506717)              | HRNet     | 88.47 (87.7) | 77.52 (74.8) | 67.95 (66.3) | 77.98 (76.27) |
| [GaitGraph2](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper)  |HRNet | 83.60 (82.0) | 72.80 (73.2) | 67.01 (63.6) | 74.47 (72.93) |
|                     [GaitTR]()   |        SimCC           |      95.02 (96.0) | 90.70 (91.3) | 89.67 (90.0) | 91.80 (92.4)          |



### [OUMVLP-Pose](https://ieeexplore.ieee.org/abstract/document/9139355/)

|                       Model                      |  Pose Estimator   |  Rank-1 (original format) 
| :------------------------------------------------: | :------: | :------: 
|                   [GaitGraph1](https://ieeexplore.ieee.org/document/9506717)                   | AlphaPose| 51.24         
|                   [GaitGraph2](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper)                  |AlphaPose| 64.53 
|                     [GaitTR](https://arxiv.org/abs/2204.03873)                     |AlphaPose|   43.61    





### [GREW](http://openaccess.thecvf.com/content/ICCV2021/html/Zhu_Gait_Recognition_in_the_Wild_A_Benchmark_ICCV_2021_paper.html)

|                       Model                       | Pose Estimator  | Rank-1  |
| :------------------------------------------------:| :-----: | :-----: |
|                   [GaitGraph1](https://ieeexplore.ieee.org/document/9506717)                  |HRNet |     36.08    |
|                   [GaitGraph2](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper)                  |HRNet |   44.41    |
|                     [GaitTR](https://arxiv.org/abs/2204.03873)                    |HRNet |   55.33   |



### [Gait3D](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Gait_Recognition_in_the_Wild_With_Dense_3D_Representations_and_CVPR_2022_paper.html)


|                       Model                       |Pose Estimator | Rank-1  |
| :------------------------------------------------:| :-----: | :-----: |
|                   [GaitGraph1](https://ieeexplore.ieee.org/document/9506717)                  | HRNet |   14.60    |
|                   [GaitGraph2](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper)                 | HRNet |   12.50      |
|                     [GaitTR](https://arxiv.org/abs/2204.03873)                  | HRNet  |   9.70    |



The results in the parentheses are mentioned in the papers. 

**Note:**
* We sincerely thank all authors for their excellent works.
* We modified the training strategy of GaitGraph1/GaitGraph2 and did not select the best model during training for the initialization of the second phase. Please refer to [configs](../configs) for the details.

