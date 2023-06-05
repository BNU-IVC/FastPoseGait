import math
import torch.nn as nn
from ..other_modules import Mish

class UnitConv2D(nn.Module):
    '''
    This class is used in GaitTR[TCN_ST] block.
    '''

    def __init__(self, D_in, D_out, kernel_size=9, stride=1, dropout=0.1, bias=True):
        super(UnitConv2D,self).__init__()
        pad = int((kernel_size-1)/2)
        self.conv = nn.Conv2d(D_in,D_out,kernel_size=(kernel_size,1)
                            ,padding=(pad,0),stride=(stride,1),bias=bias)
        self.bn = nn.BatchNorm2d(D_out)
        self.relu = Mish()
        self.dropout = nn.Dropout(dropout, inplace=False)
        #initalize
        self.conv_init(self.conv)

    def forward(self,x):
        x = self.dropout(x)
        x = self.bn(self.relu(self.conv(x)))
        return x

    def conv_init(self,module):
        n = module.out_channels
        for k in module.kernel_size:
            n = n*k
        module.weight.data.normal_(0, math.sqrt(2. / n))