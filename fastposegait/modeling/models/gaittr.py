import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base_model import BaseModel
from ..components import TCN_ST
from ..graph import Graph
import numpy as np

class GaitTR(BaseModel):

    def build_network(self, model_cfg):

        in_c = model_cfg['in_channels']
        self.num_class = model_cfg['num_class']
        self.joint_format = model_cfg['joint_format']
        self.graph = Graph(joint_format=self.joint_format,max_hop=3)

        #### Network Define ####

        # ajaceny matrix
        self.A = torch.from_numpy(self.graph.A.astype(np.float32))

        #data normalization
        num_point = self.A.shape[-1]
        self.data_bn = nn.BatchNorm1d(in_c[0] * num_point)
        
        #backbone
        backbone = []
        for i in range(len(in_c)-1):
            backbone.append(TCN_ST(in_channel= in_c[i],out_channel= in_c[i+1],A=self.A,num_point=num_point))
        self.backbone = nn.ModuleList(backbone)

        self.fcn = nn.Conv1d(in_c[-1], self.num_class, kernel_size=1)

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        x= ipts[0] 
        pose = x

        N, C, T, V, M = x.size()
        if len(x.size()) == 4:
            x = x.unsqueeze(1)
        del ipts

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)

        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                N * M, C, T, V)
        #backbone
        for _,m in enumerate(self.backbone):
            x = m(x)
        # V pooling
        x = F.avg_pool2d(x, kernel_size=(1,V))
        #M pooling
        c = x.size(1)
        t = x.size(2)
        x = x.view(N, M, c, t).mean(dim=1).view(N, c, t)#[n,c,t]
        # T pooling
        x = F.avg_pool1d(x, kernel_size=x.size()[2]) #[n,c]
        # C fcn
        x = self.fcn(x) #[n,c']
        x = F.avg_pool1d(x, x.size()[2:]) # [n,c']
        x = x.view(N, self.num_class) # n,c
        embed = x.unsqueeze(-1) # n,c,1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': labs}
            },
            'visual_summary': {
                'image/pose': pose.view(N*T, M, V, C)
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval
