import torch
import torch.nn as nn
import logging
from .units import UnitConv2D, STModule, PAGCN, PAGCN_Plus
from .units.unit_tcn import Temporal_Basic_Block, Temporal_Bottleneck_Block
from .units.unit_sgcn import Spatial_Basic_Block, Spatial_Bottleneck_Block

class TCN_ST(nn.Module):
    """
    Block of GaitTR：https://arxiv.org/pdf/2204.03873.pdf
    TCN: Temporal Convolution Network
    ST: Sptail Temporal Graph Convolution Network
    """
    def __init__(self,in_channel,out_channel,A,num_point):
        super(TCN_ST, self).__init__()
        #params
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.A = A
        self.num_point = num_point
        #network
        self.tcn = UnitConv2D(D_in=self.in_channel,D_out=self.in_channel,kernel_size=9)
        self.st = STModule(in_channels=self.in_channel,out_channels=self.out_channel,incidence=self.A,num_point=self.num_point)
        self.residual = lambda x: x
        if (in_channel != out_channel):
            self.residual_s = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1),
                nn.BatchNorm2d(out_channel),
            )
            self.down = UnitConv2D(D_in=self.in_channel,D_out=out_channel,kernel_size=1,dropout=0)
        else:
            self.residual_s = lambda x: x
            self.down = None

    def forward(self,x):
        x0 = self.tcn(x) + self.residual(x)
        y = self.st(x0) + self.residual_s(x0)
        # skip residual
        y = y + (x if(self.down is None) else self.down(x))
        return y


class ResGCN_Module(nn.Module):
    """
        ResGCN_Module
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, in_channels, out_channels, block, A, stride=1, kernel_size=[9,2],reduction=4, get_res=False,is_main=False):
        super(ResGCN_Module, self).__init__()

        if not len(kernel_size) == 2:
            logging.info('')
            logging.error('Error: Please check whether len(kernel_size) == 2')
            raise ValueError()
        if not kernel_size[0] % 2 == 1:
            logging.info('')
            logging.error('Error: Please check whether kernel_size[0] % 2 == 1')
            raise ValueError()
        temporal_window_size, max_graph_distance = kernel_size

        if  block == 'initial':
            module_res, block_res = False, False
        elif block == 'Basic':
            module_res, block_res = True, False
        else:
            module_res, block_res = False, True

        if not module_res:
            self.residual = lambda x: 0
        elif stride == 1 and in_channels == out_channels:
            self.residual = lambda x: x
        else:
            # stride =2
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, (stride,1)),
                nn.BatchNorm2d(out_channels),
            )
        
        if block in ['Basic','initial']:
            spatial_block = Spatial_Basic_Block
            temporal_block = Temporal_Basic_Block
        if block == 'Bottleneck':
            spatial_block = Spatial_Bottleneck_Block
            temporal_block = Temporal_Bottleneck_Block
        self.scn = spatial_block(in_channels, out_channels, max_graph_distance, block_res,reduction)
        if in_channels == out_channels and is_main:
            tcn_stride =True
        else:
            tcn_stride = False
        self.tcn = temporal_block(out_channels, temporal_window_size, stride, block_res,reduction,get_res=get_res,tcn_stride=tcn_stride)
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x, A):
        A = A.cuda(x.get_device())
        return self.tcn(self.scn(x, A*self.edge), self.residual(x))


class Part_AGCN_Residual(nn.Module):
    '''
    Block of GPGait: https://arxiv.org/abs/2303.05234
    '''
    def __init__(self,in_channels,out_channels,A,joint_format='coco'):
        super(Part_AGCN_Residual,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A = A
        self.asg = PAGCN(in_channels=self.in_channels,out_channels=self.out_channels,A=self.A,joint_format=joint_format)
        if (in_channels != out_channels):
            self.residual_s = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.residual_s = lambda x: x

    def forward(self, x, A, part=None):
        x = self.asg(x, A, part) + self.residual_s(x)
        return x


class Part_AGCN_TCN(nn.Module):
    def __init__(self, in_channels, out_channels, A, joint_format='coco', mask=True):
        super(Part_AGCN_TCN,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.A = A
        self.mask=mask
        out_asg = out_channels // 2
        out_res = out_channels - out_asg
        out_tcn = out_channels // 3 + 1
        out_res_s = out_channels // 3
        out_down = out_channels - out_res_s - out_tcn
        self.asg = PAGCN_Plus(in_channels=self.in_channels,out_channels=out_asg,A=self.A,joint_format=joint_format)
        self.tcn = UnitConv2D(D_in=self.out_channels, D_out=out_tcn, kernel_size=9)

        self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_res, 1),
                nn.BatchNorm2d(out_res),
            )
        self.residual_s = nn.Sequential(
                nn.Conv2d(self.out_channels, out_res_s, 1),
                nn.BatchNorm2d(out_res_s),
            )
        self.down = UnitConv2D(D_in=self.in_channels, D_out=out_down, kernel_size=1, dropout=0)

    def forward(self, x, A, part=None):
        if not self.mask:
            part = None
        x0 = torch.cat([self.asg(x, A, part), self.residual(x)], dim=1)
        y = torch.cat([self.tcn(x0), self.residual_s(x0), self.down(x)], dim=1)
        return y


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return x