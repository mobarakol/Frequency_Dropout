'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fd_filters import get_gabor_filter, get_gaussian_filter, get_laplacian_gaussian_filter
import random


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            in_planes,
            planes,
            stride=1,
        ):
        super(BasicBlock, self).__init__()
        
        self.planes = planes

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut_kernel = True
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    
    def get_new_kernels(self, ksize, freq_min, freq_max, dropout_p, filter_func=None):
        sigma = torch.tensor(np.random.uniform(freq_min, freq_max, self.planes))
        dropout = nn.Dropout(dropout_p)
        sigma = dropout(sigma)*(1-dropout_p)
        self.kernel1 = filter_func(ksize=ksize, sigma=sigma,  channels=self.planes)
        self.kernel2 = filter_func(ksize=ksize, sigma=sigma,  channels=self.planes)

    def get_new_kernels_cbs(self, kernel_size, std):
        self.kernel1 = get_gaussian_filter(ksize=kernel_size, sigma=std,channels=self.planes)
        self.kernel2 = get_gaussian_filter(ksize=kernel_size, sigma=std,channels=self.planes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(self.kernel1(out)))         

        out = self.conv2(out)
        out = self.bn2(self.kernel2(out))

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, args):
        super(ResNet, self).__init__()
        
        self.in_planes = 64
        self.in_ch = 64
        self.kernel_size = args.kernel_size
        self.filter_all = [get_gaussian_filter, get_laplacian_gaussian_filter, get_gabor_filter]
        self.num_filters = 0 if args.use_gf else len(self.filter_all) - 1
        
        self.freq_max_all = args.freq_max_all
        self.freq_min_all = args.freq_min_all
        print('dropout_p_all:',args.dropout_p_all, '\n freq_min_all',args.freq_min_all, '\n freq_max_all',
                args.freq_max_all, '\n num of filters:', self.num_filters + 1)
        self.dropout_p_all = args.dropout_p_all
        self.std = args.std
        if args.dataset == 'imagenet':
            self.conv1 = nn.Conv2d(args.in_dim, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.conv1 = nn.Conv2d(args.in_dim, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.avgpool = nn.AvgPool2d(4)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, args.num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(self.kernel1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


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
            child.get_new_kernels(self.kernel_size, freq_min, freq_max, dropout_p, filter_func)

        for child in self.layer3.children():
            child.get_new_kernels(self.kernel_size, freq_min, freq_max, dropout_p, filter_func)

        for child in self.layer4.children():
            child.get_new_kernels(self.kernel_size, freq_min, freq_max, dropout_p, filter_func)

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

        for child in self.layer4.children():
            child.get_new_kernels_cbs(self.kernel_size, self.std)


def ResNet18(args):
    return ResNet(BasicBlock, [2,2,2,2], args)

def ResNet34(args):
    return ResNet(BasicBlock, [3,4,6,3], args)

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3], args)

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3], args)



def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

