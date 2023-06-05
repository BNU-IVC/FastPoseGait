# Prepare dataset
Suppose you have downloaded the original dataset, we need to preprocess the data and save it as a pickle file. Remember to set your path to the root of processed dataset in [configs/*.yaml](configs/).

## Preprocess
**CASIA-B** 
- Step1: Download the dataset
- Step2: Unzip the dataset

**OUMVLP** 

- Step1: Download the dataset

- Step2: Unzip the dataset, run 
```
python misc/pretreamt_oumvlp_pose.py --datasetdir=<YOUR OUMVLP DATASET PATH> --dstdir=<YOUR TARGET PATH>
```
- Step3: Transform 18 keypoints to 17 keypoints (To COCO keypoints format)
```
python misc/oumvlp17.py --datasetdir=<YOUR OUMVLP DATASET PATH> --dstdir=<YOUR TARGET PATH>
```

- Processed
    ```
   OUMVLP
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
 
    ```


**GREW** 

- Step1: Download the data

- Step2: Unzip the dataset, run :
```
python pretreatment_grew_pose.py --datasetdir=<YOUR OUMVLP DATASET PATH> --dstdir=<YOUR TARGET PATH>
```
- Processed
```
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
            
```

**Gait3D**
- Step1: Download the data
- Step2: Unzip the dataset, and transform json to pickle file, run 
```
python pretreatment_gait3d_pose.py --datasetdir=<YOUR OUMVLP DATASET PATH> --dstdir=<YOUR TARGET PATH>
```
- Processed
```
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
            output data format
            ├── 2D_Poses-pkl
            │  ├── 0000
            │     ├── camid0_videoid2
            │        ├── seq0
            │           └──seq0.pkl   
```
## Split dataset
You can use the partition file in dataset folder directly, or you can create yours. Remember to set your path to the partition file in [configs/*.yaml](configs/).
