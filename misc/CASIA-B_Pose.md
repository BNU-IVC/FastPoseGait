# CASIA-B Pose

CASIA-B Pose can be downloaded in [this link](https://www.scidb.cn/en/detail?dataSetId=8ec62efd66a544939e821edeccc1f35c).

1. File Structure

```
CASIA-B-Pose
    001 (subject)
        bg-01 (type)
                000 (view)
                    mapping.txt (mapping relations between pose and RGB frames)
                    000.pkl (contains all frames)
            ......
        ......
    ......
```

2. Tensor in *.PKL

The shape of the tensor in one *.pkl file is `[T,V,C]`, where `T` represents the number of frames, `V` represents the number of keypoints, and `C` represents dimensions of each keypoint, which includes the x-coordinate, y-coordinate, and confidence score.

The order of 17 keypoints is as follows:

```
 keypoints = {
             0: "nose",  
             1: "neck"
             2: "Rshoulder"
             3: "Relbow"
             4: "Rwrist"
             5: "Lshoudler"
             6: "Lelbow"
             7: "Lwrist"
             8: "Rhip
             9: "Rknee"
             10: "Rankle"
             11: "Lhip"
             12: "Lknee"
             13: "Lankle"
             14: "Reye"
             15: "Leye"
             16: "Rear"
             17: "Lear"
         }
```