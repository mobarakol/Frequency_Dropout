"""resnext in pytorch
[1] Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He.
    Aggregated Residual Transformations for Deep Neural Networks
    https://arxiv.org/abs/1611.05431
"""

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from fd_filters import get_gabor_filter, get_gaussian_filter, get_laplacian_gaussian_filter

import random

#only implements ResNext bottleneck c


#"""This strategy exposes a new dimension, which we call “cardinality”
#(the size of the set of transformations), as an essential factor
#in addition to the dimensions of depth and width."""
CARDINALITY = 32
DEPTH = 4
BASEWIDTH = 64

#"""The grouped convolutional layer in Fig. 3(c) performs 32 groups
#of convolutions whose input and output channels are 4-dimensional.
#The grouped convolutional layer concatenates them as the outputs
#of the layer."""

class ResNextBottleNeckC(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        C = CARDINALITY #How many groups a feature map was splitted into

        #"""We note that the input/output width of the template is fixed as
        #256-d (Fig. 3), We note that the input/output width of the template
        #is fixed as 256-d (Fig. 3), and all widths are dou- bled each time
        #when the feature map is subsampled (see Table 1)."""
        D = int(DEPTH * out_channels / BASEWIDTH) #number of channels per group
        self.planes = C * D
        self.out_channels = out_channels * 4
        self.conv1 = nn.Conv2d(in_channels, C * D, kernel_size=1, groups=C, bias=False)
        self.bn1 = nn.BatchNorm2d(C * D)
        self.conv2 = nn.Conv2d(C * D, C * D, kernel_size=3, stride=stride, groups=C, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(C * D)
        self.conv3 = nn.Conv2d(C * D, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )

    def get_new_kernels(self, ksize, freq_min, freq_max, dropout_p, filter_func=None):
        sigma = torch.tensor(np.random.uniform(freq_min, freq_max, self.planes))
        dropout = nn.Dropout(dropout_p)
        sigma = dropout(sigma)*(1-dropout_p)
        self.kernel1 = filter_func(ksize=ksize, sigma=sigma,  channels=self.planes)
        sigma = torch.tensor(np.random.uniform(freq_min, freq_max, self.out_channels))
        sigma = dropout(sigma)*(1-dropout_p)
        self.kernel3 = filter_func(ksize=ksize, sigma=sigma,  channels=self.out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(self.kernel1(out)))
        out = self.conv2(out)
        out = F.relu(self.bn2(self.kernel1(out))) 
        out = self.conv3(out)
        out = self.bn3(self.kernel3(out))        
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        
class ResNext(nn.Module):

    def __init__(self, block, num_blocks, args):
        super().__init__()
        self.in_channels = 64

        self.in_ch = self.in_channels
        self.kernel_size = args.kernel_size
        self.filter_all = [get_gaussian_filter, get_laplacian_gaussian_filter, get_gabor_filter]
        self.num_filters = 0 if args.use_gf else len(self.filter_all) - 1
        self.freq_max_all = args.freq_max_all
        self.freq_min_all = args.freq_min_all
        print('dropout_p_all:',args.dropout_p_all, '\n freq_min_all',args.freq_min_all, '\n freq_max_all',
                args.freq_max_all, '\n num of filters:', self.num_filters + 1)
        self.dropout_p_all = args.dropout_p_all

        self.conv1 = nn.Conv2d(args.in_dim, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, num_blocks[0], 64, 1)
        self.layer2 = self._make_layer(block, num_blocks[1], 128, 2)
        self.layer3 = self._make_layer(block, num_blocks[2], 256, 2)
        self.layer4 = self._make_layer(block, num_blocks[3], 512, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, args.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(self.kernel1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _make_layer(self, block, num_block, out_channels, stride):
        """Building resnext block
        Args:
            block: block type(default resnext bottleneck c)
            num_block: number of blocks per layer
            out_channels: output channels per block
            stride: block stride
        Returns:
            a resnext layer
        """
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

    def get_new_kernels(self):
        f_idx = random.randint(0, self.num_filters)
        filter_func, dropout_p = self.filter_all[f_idx], self.dropout_p_all[f_idx]
        freq_min, freq_max = self.freq_min_all[f_idx], self.freq_max_all[f_idx]
        sigma = torch.tensor(np.random.uniform(freq_min,freq_max, self.in_ch))
        dropout = nn.Dropout(dropout_p)
        sigma = dropout(sigma)*(1-dropout_p)
        self.kernel1 = filter_func(ksize=self.kernel_size, sigma=sigma, channels=self.in_ch)
        for child in self.layer1.children():
            child.get_new_kernels(self.kernel_size, freq_min, freq_max, dropout_p, filter_func)

        for child in self.layer2.children():
            child.get_new_kernels(self.kernel_size, freq_min, freq_max, 1, filter_func)

        for child in self.layer3.children():
            child.get_new_kernels(self.kernel_size, freq_min, freq_max, 1, filter_func)

        for child in self.layer4.children():
            child.get_new_kernels(self.kernel_size, freq_min, freq_max, 1, filter_func)

def resnext50(args):
    """ return a resnext50(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 6, 3], args)

def resnext101():
    """ return a resnext101(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 23, 3])

def resnext152():
    """ return a resnext101(c32x4d) network
    """
    return ResNext(ResNextBottleNeckC, [3, 4, 36, 3])
