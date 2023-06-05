import torch.nn as nn

class Temporal_Basic_Block(nn.Module):
    """
        TemporalConv_Res_Block
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, channels, temporal_window_size, stride=1, residual=False,reduction=0,get_res=False,tcn_stride=False):
        super(Temporal_Basic_Block, self).__init__()

        padding = ((temporal_window_size - 1) // 2, 0)

        if not residual:
            self.residual = lambda x: 0
        elif stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (stride,1)),
                nn.BatchNorm2d(channels),
            )

        self.conv = nn.Conv2d(channels, channels, (temporal_window_size,1), (stride,1), padding)
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x + res_block + res_module)

        return x

class Temporal_Bottleneck_Block(nn.Module):
    """
        TemporalConv_Res_Bottleneck
        Arxiv: https://arxiv.org/abs/2010.09978
        Github: https://github.com/Thomas-yx/ResGCNv1
    """
    def __init__(self, channels, temporal_window_size, stride=1, residual=False, reduction=4,get_res=False, tcn_stride=False):
        super(Temporal_Bottleneck_Block, self).__init__()
        tcn_stride =False
        padding = ((temporal_window_size - 1) // 2, 0)
        inter_channels = channels // reduction
        if get_res:
            if tcn_stride:
                stride =2
            self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, 1, (2,1)),
                nn.BatchNorm2d(channels),
            )
            tcn_stride= True
        else:
            if not residual:
                self.residual = lambda x: 0
            elif stride == 1:
                self.residual = lambda x: x
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(channels, channels, 1, (2,1)),
                    nn.BatchNorm2d(channels),
                )
                tcn_stride= True

        self.conv_down = nn.Conv2d(channels, inter_channels, 1)
        self.bn_down = nn.BatchNorm2d(inter_channels)
        if tcn_stride:
            stride=2
        self.conv = nn.Conv2d(inter_channels, inter_channels, (temporal_window_size,1), (stride,1), padding)
        self.bn = nn.BatchNorm2d(inter_channels)
        self.conv_up = nn.Conv2d(inter_channels, channels, 1)
        self.bn_up = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, res_module):

        res_block = self.residual(x)

        x = self.conv_down(x)
        x = self.bn_down(x)
        x = self.relu(x)

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv_up(x)
        x = self.bn_up(x)
        x = self.relu(x + res_block + res_module)

        return x