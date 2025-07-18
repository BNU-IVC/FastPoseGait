<div align="center"><img src="resources\logo.png"  alt="logo"  width = "428" height = "250" /></div>


**FastPoseGait** is a user-friendly and flexible repository that aims to help researchers get started on **pose-based gait recognition** quickly. 
This repository is provided by [BNU-IVC](https://github.com/BNU-IVC) and supported in part by [WATRIX.AI](http://www.watrix.ai).

<div align="center">
 
[![Technical Report](https://img.shields.io/badge/Technical_Report-PDF-red.svg?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2309.00794)
[![GPGait](https://img.shields.io/badge/GPGait-PDF-red.svg?logo=arXiv&logoColor=red)](https://arxiv.org/abs/2303.05234)
[![made-with-pytorch](https://img.shields.io/badge/Made%20with-PyTorch-brightgreen)](https://pytorch.org/)
[![Percentage of issues still open](http://isitmaintained.com/badge/open/BNU-IVC/FastPoseGait.svg)](http://isitmaintained.com/project/BNU-IVC/FastPoseGait "Percentage of issues still open")

</div>

## News!
* [Jun 2025] 🔥🔥🔥 Our paper [From FastPoseGait to GPGait++: Bridging the Past and Future for Pose-based Gait Recognition](https://ieeexplore.ieee.org/document/11029177) is accepted by **TPAMI 2025**! Checkout [paper](https://ieeexplore.ieee.org/document/11029177) and [code](https://github.com/BNU-IVC/FastPoseGait/tree/main/configs/gpgait++).
* [Jul 2024] 🔥🔥🔥 Our paper [Cut out the Middleman: Revisiting Pose-based Gait Recognition](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04501.pdf) is accepted by **ECCV 2024**! Checkout [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04501.pdf) and [code](https://github.com/BNU-IVC/FastPoseGait/tree/main/configs/gaitheat).
* [Nov 2023] [SUSTech1K](https://lidargait.github.io/) and [CCPG](https://github.com/BNU-IVC/CCPG) have been supported in our project! The result of GPGait on these two benchmarks can be found in [Model Zoo](docs/model_zoo.md).
* [Sep 2023] Our technical report [FastPoseGait: A Toolbox and Benchmark for Efficient Pose-based Gait Recognition](https://arxiv.org/abs/2309.00794) and the code of <i>Improved Version</i> are released! Check out [Model Zoo](docs/model_zoo.md).
* [Jul 2023] 🔥🔥🔥  Our paper [GPGait: Generalized Pose-based Gait Recognition](https://arxiv.org/abs/2303.05234) is accepted by **ICCV 2023**! Checkout [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Fu_GPGait_Generalized_Pose-based_Gait_Recognition_ICCV_2023_paper.pdf), [poster](https://drive.google.com/file/d/196ZGrlKevgMjuXCsQvrbNnzZ6b5mTf3s/view?usp=sharing) and video([bilibili](https://www.bilibili.com/video/BV1tN411b7Qz
) [YouTube](https://www.youtube.com/watch?v=NY_MzAxpm94)).

## Supports

### Supported Algorithms
- [x] [GPGait (ICCV 2023)](https://arxiv.org/abs/2303.05234)

- [x] [GaitTR (Expert-Systems 2023)](https://onlinelibrary.wiley.com/doi/abs/10.1111/exsy.13244)

- [x] [GaitGraph2 (CVPRW 2022)](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper)

- [x] [GaitGraph1 (ICIP 2021)](https://ieeexplore.ieee.org/document/9506717)

### Supported Datasets

- [x] [CASIA-B (ICPR 2006)](https://ieeexplore.ieee.org/abstract/document/1699873/)

- [x] [OUMVLP-Pose (TBIOM 2020)](https://ieeexplore.ieee.org/abstract/document/9139355/)

- [x] [GREW (ICCV 2021)](http://openaccess.thecvf.com/content/ICCV2021/html/Zhu_Gait_Recognition_in_the_Wild_A_Benchmark_ICCV_2021_paper.html)

- [x] [Gait3D (CVPR 2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Gait_Recognition_in_the_Wild_With_Dense_3D_Representations_and_CVPR_2022_paper.html)

- [x] [SUSTech1K (CVPR 2023)](https://lidargait.github.io/)

- [x] [CCPG (CVPR 2023)](https://github.com/BNU-IVC/CCPG)

## TODO
- [x] Release the training code of GPGait++
- [x] Release the training code of GaitHeat
- [ ] Release model ckpts

## Getting Started

### For the basic usage of FastPoseGait
```
git clone https://github.com/BNU-IVC/FastPoseGait
cd FastPoseGait
```
#### 1. Installation
* python >= 3.9
* torch >= 1.8
* tqdm
* pyyaml
* tensorboard
* pytorch_metric_learning


Install the dependencies with pip:
```
pip install pyyaml tqdm tensorboard pytorch_metric_learning
pip install torch==1.8 torchvision==0.9
```
Install the dependencies by [Anaconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html):
```
conda create -n fastposegait python=3.9
conda install pytorch==1.8 torchvision -c pytorch
conda install pyyaml tqdm tensorboard -c conda-forge
pip install pytorch_metric_learning
```

#### 2. Data Preparation
* CASIA-B Pose can be downloaded from [this link](https://www.scidb.cn/en/detail?dataSetId=8ec62efd66a544939e821edeccc1f35c).
* Or to obtain the official human keypoint annotations, you can apply for it:
  * [CASIA-B official site](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
  * [OUMVLP-Pose official site](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitLPPose.html)
  * [GREW official site](https://www.grew-benchmark.org/download.html)
  * [Gait3D official site](https://gait3d.github.io/#dataset)
  * [SUSTech1K official site](https://lidargait.github.io/)
  * [CCPG official site](https://github.com/BNU-IVC/CCPG)
* Suppose you have downloaded the official annotations, you need to use our [provided script](docs/process_dataset.md)  to generate the processed pickle files.



#### 3. Training & Testing

Train a model by

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gaittr/gaittr.yaml --phase train
```

- `python -m torch.distributed.launch` [DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) launch instruction.
- `--nproc_per_node` The number of gpus to use, and it must equal to the length of `CUDA_VISIBLE_DEVICES`.
- `--cfgs` The path to config file.
- `--phase` Specified as `train`.

- `--log_to_file` If specified, the terminal log will be written on disk simultaneously.

You can run commands in [dist_train.sh](dist_train.sh) to train different models.

Test a model by

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gaittr/gaittr.yaml --phase test
```

- `--phase` Specified as `test`.

You can run commands in [dist_test.sh](dist_test.sh) for testing different models.

### For developers who wish to develop based on FastPoseGait
* [Configs](docs/configs.md) 
* [Customize Data Transforms](docs/customize_data_transforms.md)
* [Implement New Models](docs/implement_new_models.md)

## Model Zoo
Results and models are available in the [model zoo](docs/model_zoo.md). [[Google Drive]](https://drive.google.com/drive/folders/1jsQ5cZcZ5YXVvzSvyEETlN6l1B7T6paD?usp=drive_link) [[百度网盘 提取码s4jj]](https://pan.baidu.com/s/15UUKcqf6LoPBEd3123i-3g?pwd=s4jj)

## Acknowledgement
* GaitGraph/GaitGraph2: [Torben Teepe](https://scholar.google.com/citations?user=TWJuTroAAAAJ&hl=zh-CN&oi=sra)
* GaitTR: [Cun Zhang](https://github.com/zhcun)
* [OpenGait Team](https://github.com/ShiqiYu/OpenGait)
* [CASIA-B Team](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
* [OUMVLP-Pose Team](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitLPPose.html)
* [GREW Team](https://www.grew-benchmark.org/download.html)
* [Gait3D Team](https://gait3d.github.io/#dataset)
* [SUSTech1K Team](https://lidargait.github.io/)
* [CCPG Team](https://github.com/BNU-IVC/CCPG)

## Citation

If you find this project useful in your research, please consider citing: 
```
@article{meng2023fastposegait,
  title={FastPoseGait: A Toolbox and Benchmark for Efficient Pose-based Gait Recognition},
  author={Meng, Shibei and Fu, Yang and Hou, Saihui and Cao, Chunshui and Liu, Xu and Huang, Yongzhen},
  journal={arXiv preprint arXiv:2309.00794},
  year={2023}
}
```


**Note**: This code is strictly intended for **academic purposes** and can not be utilized for any form of commercial use.


## Authors
This project is built and maintained by [ShiBei Meng](https://github.com/DreamShibei) and [Yang Fu](https://www.yangfu.site). 
We build this project based on the open-source project [OpenGait](https://github.com/ShiqiYu/OpenGait).

We will keep up with the latest progress of the community, and support more popular algorithms and frameworks. We also appreciate all contributions to improve FastPoseGait. If you have any feature requests, please feel free to leave a comment, file an issue or contact the authors:

* ShiBei Meng, mengshibei@mail.bnu.edu.cn
* Yang Fu, aleeyanger@gmail.com
