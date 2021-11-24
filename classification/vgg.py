import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import random

from fd_filters import get_gabor_filter, get_gaussian_filter, get_laplacian_gaussian_filter


class VGG16_conv(torch.nn.Module):
    def __init__(self, n_classes, args):
        super(VGG16_conv, self).__init__()
        
        self.ch_list = [64,128, 256, 512]
        self.kernel_size = args.kernel_size
        self.filter_all = [get_gaussian_filter, get_laplacian_gaussian_filter, get_gabor_filter]
        self.num_filters = 0 if args.use_gf else len(self.filter_all) - 1
        self.freq_max_all = args.freq_max_all
        self.freq_min_all = args.freq_min_all
        print('dropout_p_all:',args.dropout_p_all, '\n freq_min_all',args.freq_min_all, '\n freq_max_all',
                args.freq_max_all, '\n num of filters:', self.num_filters + 1)

        self.dropout_p_all = args.dropout_p_all

        self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(args.in_dim, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 64, 3, padding=1),
        )
        self.post1 = torch.nn.Sequential(
                nn.BatchNorm2d(64),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )
        self.conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 128, 3, padding=1),
        )
        self.post2 = torch.nn.Sequential(
                nn.BatchNorm2d(128),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )
        self.conv3 = torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 256, 3, padding=1),
        )
        self.post3 = torch.nn.Sequential(
                nn.BatchNorm2d(256),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )
        self.conv4 = torch.nn.Sequential(
                torch.nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, 3, padding=1),
        )
        self.post4 = torch.nn.Sequential(
                nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )
        self.conv5 = torch.nn.Sequential(
                torch.nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, 3, padding=1),
        )
        self.post5 = torch.nn.Sequential(
                nn.BatchNorm2d(512),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, stride=2)
        )

        self.classifier = torch.nn.Sequential(
                torch.nn.Linear(512, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, 4096),
                torch.nn.ReLU(),
                torch.nn.Dropout(),
                torch.nn.Linear(4096, n_classes)
        )
        
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

    def get_new_kernels(self):
        f_idx = random.randint(0, self.num_filters)
        filter_func, dropout_p = self.filter_all[f_idx], self.dropout_p_all[f_idx]
        freq_min, freq_max = self.freq_min_all[f_idx], self.freq_max_all[f_idx]
        sigma = torch.tensor(np.random.uniform(freq_min,freq_max, self.ch_list[0]))
        dropout = nn.Dropout(dropout_p)
        sigma = dropout(sigma)*(1-dropout_p)
        self.kernel1 = filter_func(ksize=self.kernel_size, sigma=sigma, channels=self.ch_list[0])

        sigma = torch.tensor(np.random.uniform(freq_min,freq_max, self.ch_list[1]))
        dropout = nn.Dropout(dropout_p)
        sigma = dropout(sigma)*(1-dropout_p)
        self.kernel2 = filter_func(ksize=self.kernel_size, sigma=sigma, channels=self.ch_list[1])

        sigma = torch.tensor(np.random.uniform(freq_min,freq_max, self.ch_list[2]))
        dropout = nn.Dropout(dropout_p)
        sigma = dropout(sigma)*(1-dropout_p)
        self.kernel3 = filter_func(ksize=self.kernel_size, sigma=sigma, channels=self.ch_list[2])

        sigma = torch.tensor(np.random.uniform(freq_min,freq_max, self.ch_list[3]))
        dropout = nn.Dropout(dropout_p)
        sigma = dropout(sigma)*(1-dropout_p)
        self.kernel4 = filter_func(ksize=self.kernel_size, sigma=sigma, channels=self.ch_list[3])
        sigma = torch.tensor(np.random.uniform(freq_min,freq_max, self.ch_list[3]))
        dropout = nn.Dropout(dropout_p)
        sigma = dropout(sigma)*(1-dropout_p)
        self.kernel5 = filter_func(ksize=self.kernel_size, sigma=sigma, channels=self.ch_list[3])

    def forward(self, x, return_intermediate=False):
        output = self.conv1(x)
        output = self.kernel1(output) 
        output = self.post1(output)

        output = self.conv2(output)
        output = self.kernel2(output) 
        output = self.post2(output)

        output = self.conv3(output)
        output = self.kernel3(output) 
        output = self.post3(output)

        output = self.conv4(output)
        output = self.kernel4(output) 
        output = self.post4(output)

        output = self.conv5(output)
        output = self.kernel5(output) 

        if return_intermediate:
            output = output.view(output.size(0), -1)
            return output

        output = self.post5(output)

        output = output.view(output.size(0), -1)
        output = self.classifier(output)

        return output
