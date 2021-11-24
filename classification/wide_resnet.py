import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import random
import sys
import numpy as np
from fd_filters import get_gabor_filter, get_gaussian_filter, get_laplacian_gaussian_filter

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

        self.planes = planes
    
    def get_new_kernels(self, ksize, freq_min, freq_max, dropout_p, filter_func=None):
        sigma = torch.tensor(np.random.uniform(freq_min, freq_max, self.planes))
        dropout = nn.Dropout(dropout_p)
        sigma = dropout(sigma)*(1-dropout_p)
        self.kernel1 = filter_func(ksize=ksize, sigma=sigma,  channels=self.planes)
        self.kernel2 = filter_func(ksize=ksize, sigma=sigma,  channels=self.planes)
    
    def get_new_kernels_cbs(self, kernel_size, std):
        self.kernel1 = get_gaussian_filter(ksize=kernel_size, sigma=std, channels=self.planes)
        self.kernel2 = get_gaussian_filter(ksize=kernel_size, sigma=std, channels=self.planes)

    def forward(self, x):
        out = self.dropout(self.kernel1(self.conv1(F.relu(self.bn1(x)))))
        out = self.kernel2(self.conv2(F.relu(self.bn2(out))))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, args):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.in_ch = nStages[0]
        self.kernel_size = args.kernel_size
        self.filter_all = [get_gaussian_filter, get_laplacian_gaussian_filter, get_gabor_filter]
        self.num_filters = 0 if args.use_gf else len(self.filter_all) - 1
        self.freq_max_all = args.freq_max_all
        self.freq_min_all = args.freq_min_all
        print('dropout_p_all:',args.dropout_p_all, '\n freq_min_all',args.freq_min_all, '\n freq_max_all',
                args.freq_max_all, '\n num of filters:', self.num_filters + 1)
        self.dropout_p_all = args.dropout_p_all
        self.std = args.std
        self.conv1 = conv3x3(args.in_dim, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def get_new_kernels(self):
        f_idx = random.randint(0, self.num_filters)
        freq_max = self.freq_max_all[f_idx]
        freq_min = self.freq_min_all[f_idx]
        dropout_p = self.dropout_p_all[f_idx]
        filter_func = self.filter_all[f_idx]
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
    
    def get_new_kernels_cbs(self, epoch_count):
        if epoch_count % 5 == 0 and epoch_count is not 0:
            self.std *= 0.9
            
        self.kernel1 = get_gaussian_filter(ksize=self.kernel_size, sigma=self.std, channels=self.in_ch)

        for child in self.layer1.children():
            child.get_new_kernels_cbs(self.kernel_size, self.std)

        for child in self.layer2.children():
            child.get_new_kernels_cbs(self.kernel_size, self.std)

        for child in self.layer3.children():
            child.get_new_kernels_cbs(self.kernel_size, self.std)


    def forward(self, x):
        out = self.kernel1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

if __name__ == '__main__':
    net = Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())
