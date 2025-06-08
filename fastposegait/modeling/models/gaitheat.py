import torch

from ..base_model import BaseModel
from ..components import SeparateBNNecks, SeparateFCs, PackSequenceWrapper, SetBlockWrapper, HorizontalPoolingPyramid

from torch.nn import functional as F
import torch.nn as nn

from einops import rearrange

class GaitHeat(BaseModel):

    def build_network(self, model_cfg):

        model_cfg['backbone_cfg']['type'] = 'ResNet9_layer1'
        self.Backbone1 = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone1 = SetBlockWrapper(self.Backbone1)

        model_cfg['backbone_cfg']['type'] = 'ResNet9_layer2'
        self.Backbone2 = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone2 = SetBlockWrapper(self.Backbone2)

        model_cfg['backbone_cfg']['type'] = 'ResNet9_layer2'
        self.Backbone2_fusion = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone2_fusion = SetBlockWrapper(self.Backbone2_fusion)

        model_cfg['backbone_cfg']['type'] = 'ResNet9_layer3'
        self.Backbone3 = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone3 = SetBlockWrapper(self.Backbone3)

        model_cfg['backbone_cfg']['type'] = 'ResNet9_layer3'
        self.Backbone3_fusion = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone3_fusion = SetBlockWrapper(self.Backbone3_fusion)

        model_cfg['backbone_cfg']['type'] = 'ResNet9_layer4'
        self.Backbone4 = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone4 = SetBlockWrapper(self.Backbone4)

        model_cfg['backbone_cfg']['type'] = 'ResNet9_layer4'
        self.Backbone4_fusion = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone4_fusion = SetBlockWrapper(self.Backbone4_fusion)

        self.FCs_part = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks_part = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

        self.FCs_global = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks_global = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

        self.FCs_fusion = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks_fusion  = SeparateBNNecks(**model_cfg['SeparateBNNecks'])

        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])


    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs

        sils = ipts[0]
        mat = ipts[1]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')
        n, c, s, h, w = sils.size()

        ### Feature Alignment ###
        mat1 = mat[...,0]
        mat3 = mat[...,1]

        mat1 = mat1.reshape(n*s,2,3) # n*s 2x3 affine matrix
        mat3 = mat3.reshape(n*s,2,3)
        #sils n c s h w
        sils = sils.permute(0,2,1,3,4).contiguous() # n s c h w
        sils = sils.reshape(n*s, c, h, w) # n*s c h w
        # trans
        grid = F.affine_grid(mat1, sils.size())
        sils = F.grid_sample(sils.half(), grid.half(),padding_mode="border")
        
        grid = F.affine_grid(mat3, sils.size())
        sils = F.grid_sample(sils.half(), grid.half(),padding_mode="border")
        #reshape
        sils = sils.reshape(n, s, c, h, w) # expand
        sils = sils.permute(0,2,1,3,4).contiguous() # [n c s h w]



        ### Feature Extraction ###
        ##input
        l1 = sils[:,0:1] # [n 1 s h w]
        l2 = sils[:,1:2] #[n 1 s h w]
        g = torch.max(sils,dim=1,keepdim=True)[0] #[n 1 s h w]
        in_1 = torch.cat([l1,l2,g],dim=0) #[3*n 1 s h w]

        ##1model
        outs1 = self.Backbone1(in_1)  # [n, c, s, h, w]
        ##1out
        fusion1 = torch.max(torch.stack([outs1[n*0:n*1],outs1[n*1:n*2],outs1[n*2:n*3]],dim=-1), dim=-1)[0] # n c s h w

        
        ##2model
        outs2 = self.Backbone2(outs1)  # [n, c, s, h, w]
        fusion2 = self.Backbone2_fusion(fusion1)
        fusion2 = torch.max(torch.stack([outs2[n*0:n*1],outs2[n*1:n*2],outs2[n*2:n*3]],dim=-1), dim=-1)[0] + fusion2

        ##3model
        outs3 = self.Backbone3(outs2)  # [n, c, s, h, w]
        fusion3 = self.Backbone3_fusion(fusion2)
        fusion3 = torch.max(torch.stack([outs3[n*0:n*1],outs3[n*1:n*2],outs3[n*2:n*3]],dim=-1), dim=-1)[0] + fusion3
    
        ##4model
        outs4 = self.Backbone4(outs3)  # [n, c, s, h, w]
        fusion4 = self.Backbone4_fusion(fusion3)
        fusion4 = torch.max(torch.stack([outs4[n*0:n*1],outs4[n*1:n*2],outs4[n*2:n*3]],dim=-1), dim=-1)[0] + fusion4

        ##PHI
        l1 = outs4[n*0:n*1]
        l2 = outs4[n*1:n*2]
        g = outs4[n*2:n*3]

        parts_branch = torch.max(torch.stack([l1 ,l2], dim=-1),dim=-1)[0] #[n c s h w]
        global_branch = g
        fusion_branch =fusion4

        # Temporal Pooling, TP
        parts_branch = self.TP(parts_branch, seqL, options={"dim": 2})[0]  # [n, c, h, w]
        global_branch = self.TP(global_branch, seqL, options={"dim": 2})[0]
        fusion_branch = self.TP(fusion_branch, seqL, options={"dim": 2})[0]

        # Horizontal Pooling Matching, HPM
        parts_feat = self.HPP(parts_branch)  # [n, c, p]
        global_feat = self.HPP(global_branch) 
        fusion_feat = self.HPP(fusion_branch)


        part_embed_1 = self.FCs_part(parts_feat)  # [n, c, p]
        part_embed_2, part_logits = self.BNNecks_part(part_embed_1)  # [n, c, p]
        part_embed = part_embed_1

        global_embed_1 = self.FCs_global(global_feat)  # [n, c, p]
        global_embed_2, global_logits = self.BNNecks_global(global_embed_1)  # [n, c, p]
        global_embed = global_embed_1

        fusion_embed_1 = self.FCs_fusion(fusion_feat)  # [n, c, p]
        fusion_embed_2, fusion_logits = self.BNNecks_fusion(fusion_embed_1)  # [n, c, p]
        fusion_embed = fusion_embed_1

        retval = {
            'training_feat': {
                'part_triplet': {'embeddings': part_embed_1, 'labels': labs},
                'part_softmax': {'logits': part_logits, 'labels': labs},
                'global_triplet': {'embeddings': global_embed_1, 'labels': labs},
                'global_softmax': {'logits': global_logits, 'labels': labs},
                'fusion_triplet': {'embeddings': fusion_embed_1, 'labels': labs},
                'fusion_softmax': {'logits': fusion_logits, 'labels': labs},
            },
            'visual_summary': {
                # 'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': fusion_embed_2
            }
        }
        return retval



