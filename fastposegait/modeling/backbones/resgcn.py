import torch
import torch.nn as nn
from ..components.blocks import ResGCN_Module

class ResGCN_Input_Branch(nn.Module):
    """
        ResGCNInputBranch_Module
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, input_branch, block, A, input_num , reduction = 4):
        super(ResGCN_Input_Branch, self).__init__()

        self.register_buffer('A', A)

        module_list = []
        for i in range(len(input_branch)-1):
            if i==0:
                module_list.append(ResGCN_Module(input_branch[i],input_branch[i+1],'initial',A, reduction=reduction))
            else:
                module_list.append(ResGCN_Module(input_branch[i],input_branch[i+1],block,A,reduction=reduction))
        

        self.bn = nn.BatchNorm2d(input_branch[0])
        self.layers = nn.ModuleList(module_list)

    def forward(self, x):

        x = self.bn(x)
        for layer in self.layers:
            x = layer(x, self.A)

        return x
    
    
class ResGCN(nn.Module):
    """
        ResGCN
        Arxiv: https://arxiv.org/abs/2010.09978
    """
    def __init__(self, input_num, input_branch, main_stream,num_class, reduction, block, graph):
        super(ResGCN, self).__init__()
        self.graph = graph
        self.head= nn.ModuleList(
            ResGCN_Input_Branch(input_branch, block, graph, input_num ,reduction)
            for _ in range(input_num)
        )
        
        main_stream_list = []
        for i in range(len(main_stream)-1):
            if main_stream[i]==main_stream[i+1]:
                stride = 1
            else:
                stride = 2
            if i ==0:
                main_stream_list.append(ResGCN_Module(main_stream[i]*input_num,main_stream[i+1],block,graph,stride=1,reduction = reduction,get_res=True,is_main=True))
            else:
                main_stream_list.append(ResGCN_Module(main_stream[i],main_stream[i+1],block,graph,stride = stride, reduction = reduction,is_main=True))
        self.backbone = nn.ModuleList(main_stream_list)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Linear(256, num_class)

    def forward(self, x):
        # input branch
        x_cat = []
        for i, branch in enumerate(self.head):
            x_cat.append(branch(x[:, i]))
        x = torch.cat(x_cat, dim=1)

        # main stream
        for layer in self.backbone:
            x = layer(x, self.graph)

        # output
        x = self.global_pooling(x)
        x = x.squeeze(-1)
        x = self.fcn(x.squeeze((-1)))
        
        return x