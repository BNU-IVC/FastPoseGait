from torch.nn import functional as F
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from ..components.blocks import BasicConv2d


block_map = {'BasicBlock': BasicBlock,
             'Bottleneck': Bottleneck}


class ResNet9_layer4(ResNet):
    def __init__(self, block, channels=[32, 64, 128, 256], in_channel=1, layers=[1, 2, 2, 1], strides=[1, 2, 2, 1], maxpool=True):
        if block in block_map.keys():
            block = block_map[block]
        else:
            raise ValueError(
                "Error type for -block-Cfg-, supported: 'BasicBlock' or 'Bottleneck'.")
        self.maxpool_flag = maxpool
        super(ResNet9_4, self).__init__(block, layers)

        # Not used #
        self.fc = None
        ############
        self.inplanes =256
        self.bn1 =  None

        self.conv1 =  None

        self.layer1 =  None

        self.layer2 =  None
        self.layer3 =  None
        self.layer4 = self._make_layer(
            block, channels[3], layers[3], stride=strides[3], dilate=False)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        if blocks >= 1:
            layer = super()._make_layer(block, planes, blocks, stride=stride, dilate=dilate)
        else:
            def layer(x): return x
        return layer

    def forward(self, x):

        x = self.layer4(x)
        return x

