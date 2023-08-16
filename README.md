<div align="center"><img src="resources\logo.png"  alt="logo"  width = "428" height = "250" /></div>


**FastPoseGait** is a user-friendly and flexible repository that aims to help researchers get started on **pose-based gait recognition** quickly. Just the pre-beta version is released now, and an enhanced version as well as a technical report will be released as soon as possible.
This repository is provided by [BNU-IVC](https://github.com/BNU-IVC) and supported in part by [WATRIX.AI](http://www.watrix.ai).

The code of [GPGait: Generalized Pose-based Gait Recognition (ICCV2023)](https://arxiv.org/abs/2303.05234) is released.

## Supports

### Supported Algorithms
- [x] [GaitTR(Arxiv2022)](https://arxiv.org/abs/2204.03873)

- [x] [GaitGraph2(CVPRW2022)](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/papers/Teepe_Towards_a_Deeper_Understanding_of_Skeleton-Based_Gait_Recognition_CVPRW_2022_paper)

- [x] [GaitGraph1(ICIP2021)](https://ieeexplore.ieee.org/document/9506717)
- [x] [GPGait(ICCV2023)](https://arxiv.org/abs/2303.05234)

### Supported Datasets

- [x] [CASIA-B(ICPR2006)](https://ieeexplore.ieee.org/abstract/document/1699873/)

- [x] [OUMVLP-Pose(TBIOM2020)](https://ieeexplore.ieee.org/abstract/document/9139355/)

- [x] [GREW(ICCV2021)](http://openaccess.thecvf.com/content/ICCV2021/html/Zhu_Gait_Recognition_in_the_Wild_A_Benchmark_ICCV_2021_paper.html)

- [x] [Gait3D(CVPR2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_Gait_Recognition_in_the_Wild_With_Dense_3D_Representations_and_CVPR_2022_paper.html)



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


Install the dependencies by pip:
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

* To obtain the human keypoints annotations, you can apply for it:
  * [CASIA-B official site](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
  * [OUMVLP-Pose official site](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitLPPose.html)
  * [GREW official site](https://www.grew-benchmark.org/download.html)
  * [Gait3D official site](https://gait3d.github.io/#dataset)
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
Results and models are available in the [model zoo](docs/model_zoo.md). [[Google Drive]](https://drive.google.com/drive/folders/1SUsy-AVZLuHcRfaLO_40SEwoqxvecuZI?usp=sharing) [[百度网盘 提取码54mk]](https://pan.baidu.com/s/1cAiqP5tpO3hN5k5KOdUu2g?pwd=54mk)

## Acknowledgement
* GaitGraph/GaitGraph2: [Torben Teepe](https://scholar.google.com/citations?user=TWJuTroAAAAJ&hl=zh-CN&oi=sra)
* GaitTR: [Cun Zhang](https://arxiv.org/abs/2204.03873)
* [OpenGait Team](https://github.com/ShiqiYu/OpenGait)
* [CASIA-B Team](http://www.cbsr.ia.ac.cn/english/Gait%20Databases.asp)
* [OUMVLP-Pose Team](http://www.am.sanken.osaka-u.ac.jp/BiometricDB/GaitLPPose.html)
* [GREW Team](https://www.grew-benchmark.org/download.html)
* [Gait3D Team](https://gait3d.github.io/#dataset)

## Citation

If you find this project useful in your research, please consider citing: 
```
@misc{fastposegait2023,
  author =       {Shibei Meng, Yang Fu, Saihui Hou, Yongzhen Huang},
  title =        {FastPoseGait},
  howpublished = {\url{https://github.com/BNU-IVC/FastPoseGait}},
  year =         {2023}
}
```
We will also release a technical report later.

**Note**: This code is strictly intended for **academic purposes** and can not be utilized for any form of commercial use.


## Authors
This project is built and maintained by [ShiBei Meng](https://github.com/DreamShibei) and [Yang Fu](https://www.yangfu.site). 
We build this project based on the open-source project [OpenGait](https://github.com/ShiqiYu/OpenGait).

We will keep up with the latest progress of the community, and support more popular algorithms and frameworks. We also appreciate all contributions to improve FastPoseGait. If you have any feature requests, please feel free to leave a comment, file an issue or contact the authors:

* ShiBei Meng, mengshibei0208@gmail.com
* Yang Fu, aleeyanger@gmail.com
