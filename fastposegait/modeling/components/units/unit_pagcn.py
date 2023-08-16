import torch
import torch.nn as nn
from torch.autograd import Variable
from ...graph import GraphPartition
from ..other_modules import Mish

class PAGCN(nn.Module):
    '''
    This class implements Part-Aware Graph Convolution used in GPGait
    Paper link: https://arxiv.org/abs/2303.05234
    '''     
    def __init__(self, in_channels, out_channels, A, joint_format, coff_embedding=4, num_subset=3):
        super(PAGCN, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(A)
        nn.init.constant_(self.PA, 1e-6)
        self.A = Variable(A, requires_grad=False)
        self.partition = GraphPartition(joint_format=joint_format)
        self.num_subset = num_subset
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = Mish()

    def Partition2Mask(self, partition,num_point):
        part_mask = torch.zeros([num_point, num_point],dtype=torch.float)
        for part in partition:
            for i in part:
                for j in part:
                    part_mask[i, j] = 1.
        part_mask = part_mask.unsqueeze(0)
        return part_mask

    def forward(self, x, A, part=None):
        N, C, T, V = x.size()
        A = A.cuda(x.get_device())
        if part:
            partition = self.partition(part)
            num_point = A.shape[-1]
            part_mask = self.Partition2Mask(partition, num_point)
            part_mask = part_mask.cuda(x.get_device())
        A = A + self.PA
        y = None
        for i in range(self.num_subset):
            q = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(N, V, self.inter_c * T)
            k = self.conv_b[i](x).view(N, self.inter_c * T, V)
            CA = self.soft(torch.matmul(q, k) / q.size(-1))  # N V V
            A1 = CA + A[i]
            if part:
                A1 = A1 * part_mask
            v = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(v, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        y = self.bn(y)
        return self.relu(y)