# GaitHeat: Revisiting Pose-based Gait Recognition

We provide instructions for preparing data and running the code.
## Data Preparation

### 1. Obtaining RGB Data

Currently supported gait datasets that provide RGB video data include:

- **CASIA-B**
- **SUSTech1K**
- **CCPG**

To obtain these datasets, please contact the dataset providers directly via email. Access may be subject to license agreements or academic use policies.

### 2. Generating Heatmap and Pose Data

After downloading and organizing the RGB datasets, you can run the following script to generate heatmap and pose data using a pose estimation model:

```bash
python misc/pretreatment_heatmap_pose.py
```

## Code
Follow the original instructions in fastposegait to train and test the model.