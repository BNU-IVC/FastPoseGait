import torch
import torch.nn as nn
import numpy as np
from ..base_model import BaseModel
from ..graph import Graph
from ..components import SeparateBNNecks, SeparateFCs, PackSequenceWrapper, SemanticPyramidPooling, Part_AGCN_TCN


class GPGait_PlusPlus(BaseModel):

    def build_network(self, model_cfg):

        #### Configs ####

        in_c = model_cfg['in_channels']
        class_num = model_cfg['num_class']
        share_num = model_cfg['share_num']
        mask = model_cfg['mask']

        self.pose = NewNet(in_c=in_c, share_num=share_num, first_c = 2, mask=mask)
        self.bone = NewNet(in_c=in_c, share_num=share_num, first_c = 2, mask=mask)
        self.angle = NewNet(in_c=in_c, share_num=share_num, first_c = 1, joint_format = 'coco-no-head', mask=mask)

        # Head
        self.head = SeparateFCs(parts_num=19, in_channels=256, out_channels=256)
        # BNNeck
        self.BNNecks = SeparateBNNecks(class_num = class_num, in_channels=256, parts_num=19)

    def forward(self, inputs):

        ipts, labs, _, _, seqL = inputs

        x = ipts[0] 
        pose = x 
        N, C, T, V, M = x.size()
        if len(x.size()) == 4:
            x = x.unsqueeze(1)
        del ipts
        x_pose = x[:,:2,...]
        x_bone = x[:,2:4,...]
        x_angle = x[:,4:5,:,5:,:]
        feature_pose = self.pose(x_pose, seqL)
        feature_bone = self.bone(x_bone, seqL)
        feature_angle = self.angle(x_angle, seqL)

        feature = torch.cat([feature_pose, feature_bone, feature_angle],-1)
        # feature = feature_pose

        #Sep Fcs
        embed_1 = self.head(feature) # [n,c,p] torch.Size([256, 256, 8])

        # BNneck
        embed_2, logits = self.BNNecks(embed_1) # torch.Size([256, 256, 8])
        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed , 'labels': labs},
                'softmax':{'logits':logits,'labels':labs}
            },
            'visual_summary': {
                'image/sils': pose.view(N*T, M, V, C)
            },
            'inference_feat': {
                'embeddings': embed_2
            }
        }
        return retval


class NewNet(nn.Module):
    def __init__(self, in_c, share_num, first_c, joint_format='coco', mask=True):
        super(NewNet,self).__init__()
        in_c = in_c
        share_num = 3
        self.joint_format = joint_format

        self.graph = Graph(joint_format)

        first_c = first_c

        #### Network Define ####

        # ajaceny matrix
        self.A = torch.from_numpy(self.graph.A.astype(np.float32))
        # data normalization
        num_point = self.A.shape[-1]
        self.data_bn = nn.BatchNorm1d(first_c * num_point)

        ## layer-share ##
        layer_share = []
        layer_share.append(Part_AGCN_TCN(in_channels= first_c, out_channels= in_c[1], A = self.A, joint_format=joint_format, mask=mask))
        for i in range(1, share_num):
            layer_share.append(Part_AGCN_TCN(in_channels= in_c[i], out_channels= in_c[i+1], A = self.A, joint_format=joint_format, mask=mask))
        self.layer_share = nn.ModuleList(layer_share)

        ## global backbone ##
        backbone = []
        for i in range(share_num, len(in_c)-1):
            backbone.append(Part_AGCN_TCN(in_channels = in_c[i], out_channels = in_c[i+1], A = self.A, joint_format=joint_format, mask=mask))
        self.backbone = nn.ModuleList(backbone)

        ## part backbone ##
        part_2 = []
        for i in range(share_num, len(in_c)-1):
            part_2.append(Part_AGCN_TCN(in_channels = in_c[i], out_channels = in_c[i+1], A = self.A, joint_format=joint_format, mask=mask))
        self.part_2 = nn.ModuleList(part_2)

        if joint_format == 'coco':
            part_2_2 = []
            for i in range(share_num, len(in_c)-1):
                part_2_2.append(Part_AGCN_TCN(in_channels = in_c[i], out_channels = in_c[i+1], A = self.A, joint_format=joint_format, mask=mask))
            self.part_2_2 = nn.ModuleList(part_2_2)

            part_3_1 = []
            for i in range(share_num, len(in_c)-1):
                part_3_1.append(Part_AGCN_TCN(in_channels = in_c[i], out_channels = in_c[i+1], A = self.A, joint_format=joint_format, mask=mask))
            self.part_3_1 = nn.ModuleList(part_3_1)

        # set pooling
        self.set_pooling = PackSequenceWrapper(torch.max)

        # SPP
        self.spp = SemanticPyramidPooling(joint_format=self.joint_format)

   
    def forward(self, x, seqL):

        #### Network Begin ####

        # ajaceny matrix
        graph = Graph(self.joint_format)
        global_A = torch.from_numpy(graph.A.astype(np.float32))

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)

        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(
                N * M, C, T, V)

        x_share = x

        for i,m in enumerate(self.layer_share):
            x_share = m(x_share, A=global_A, part=1)

        ## part backbone ##
        x_multi_brch = []

        # 2-part-1 #
        x_this_part = x_share.clone()
        for i,m in enumerate(self.part_2):
            x_this_part = m(x_this_part, A=global_A, part=2)
        # Set Pooling
        x_this_part = self.set_pooling(x_this_part,seqL,options={"dim":2})[0]
        # V Pooling
        temp = self.spp(x_this_part, part=2)
        x_multi_brch.append(temp)

        if self.joint_format == 'coco':
            # 2-part-2 #
            x_this_part = x_share.clone()
            for i,m in enumerate(self.part_2_2):
                x_this_part = m(x_this_part, A=global_A, part=3)
            # Set Pooling
            x_this_part = self.set_pooling(x_this_part,seqL,options={"dim":2})[0]
            # V Pooling
            temp = self.spp(x_this_part, part=3)
            x_multi_brch.append(temp)

            # 3-part-1 #
            x_this_part = x_share.clone()
            for i,m in enumerate(self.part_3_1):
                x_this_part = m(x_this_part, A=global_A, part=4)
            # Set Pooling
            x_this_part = self.set_pooling(x_this_part,seqL,options={"dim":2})[0]
            # V Pooling
            temp = self.spp(x_this_part, part=4)
            x_multi_brch.append(temp)

        ## global backbone ##
        x_global = x_share.clone()
        for i,m in enumerate(self.backbone):
            x_global = m(x_global, A=global_A)
        # Set Pooling
        x_global = self.set_pooling(x_global,seqL,options={"dim":2})[0]
        # V Pooling
        temp = self.spp(x_global)
        x_multi_brch.append(temp)

        x_p = torch.cat(x_multi_brch, -1) # torch.Size([256, 256, 8])

        return x_p