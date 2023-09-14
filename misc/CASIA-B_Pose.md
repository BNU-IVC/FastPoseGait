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
keypoints =
    {
        1: "nose",
        2: "left_eye",
        3: "right_eye",
        4: "left_ear",
        5: "right_ear",
        6: "left_shoulder",
        7: "right_shoulder",
        8: "left_elbow",
        9: "right_elbow",
        10: "left_wrist",
        11: "right_wrist",
        12: "left_hip",
        13: "right_hip",
        14: "left_knee",
        15: "right_knee",
        16: "left_ankle",
        17: "right_ankle"
    }
```
