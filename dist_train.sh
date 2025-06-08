#!/usr/bin/env bash
export MASTER_PORT=$((12000 + $RANDOM % 20000))


# # **************** For CASIA-B ****************

# GaitGraph1
## phase1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gaitgraph1/gaitgraph1_phase1_plus.yaml --phase train --log_to_file

# GaitGraph2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gaitgraph2/gaitgraph2_plus.yaml --phase train --log_to_file

# GaitTR
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gaittr/gaittr_plus.yaml --phase train --log_to_file 

# GPGait
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gpgait++/gpgait++.yaml --phase train --log_to_file 

# # **************** For OUMVLP ****************

# GaitGraph1
## phase1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gaitgraph1/gaitgraph1_phase1_OUMVLP_plus.yaml --phase train --log_to_file

# GaitGraph2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gaitgraph2/gaitgraph2_OUMVLP_plus.yaml --phase train --log_to_file

# GaitTR
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gaittr/gaittr_OUMVLP_plus.yaml --phase train --log_to_file 

#GPGait
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gpgait/gpgait_OUMVLP.yaml --phase train --log_to_file 


# # **************** For Gait3D ****************


# GaitGraph1
## phase1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gaitgraph1/gaitgraph1_phase1_Gait3D_plus.yaml --phase train --log_to_file

# GaitGraph2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gaitgraph2/gaitgraph2_Gait3D_plus.yaml --phase train --log_to_file

# GaitTR
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gaittr/gaittr_Gait3D_plus.yaml --phase train --log_to_file 

# GPGait
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gpgait/gpgait_Gait3D.yaml --phase train --log_to_file 


# # **************** For GREW ****************

# GaitGraph1
## phase1
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gaitgraph1/gaitgraph1_phase1_GREW_plus.yaml --phase train --log_to_file

# GaitGraph2
# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gaitgraph2/gaitgraph2_GREW_plus.yaml --phase train --log_to_file

# GaitTR
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gaittr/gaittr_GREW_plus.yaml --phase train --log_to_file 

# GPGait
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --master_port=$MASTER_PORT --nproc_per_node=4 fastposegait/main.py --cfgs ./configs/gpgait/gpgait_GREW.yaml --phase train --log_to_file 