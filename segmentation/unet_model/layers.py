from collections import OrderedDict

import torch
from torch import nn, nn as nn

import numpy as np
import math

#####Gaussian#####
def get_gaussian_kernel_3d(ksize=0, sigma=0, channels=1):
    x_coord = torch.arange(ksize)
    x_grid_2d = x_coord.repeat(ksize).view(ksize, ksize)
    x_grid = x_coord.repeat(ksize*ksize).view(ksize, ksize, ksize)
    y_grid_2d = x_grid_2d.t()
    y_grid  = y_grid_2d.repeat(ksize,1).view(ksize, ksize, ksize)
    z_grid = y_grid_2d.repeat(1,ksize).view(ksize, ksize, ksize)
    xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()
    mean = (ksize - 1)/2.
    #variance = torch.tensor(sigma**2.)
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance.view(channels,1,1,1) + 1e-16)) *\
        torch.exp( -torch.sum((xyz_grid - mean)**2., dim=-1).view(1, ksize, ksize, ksize).repeat(channels,1,1,1) /\
        (2*variance.view(channels,1,1,1) + 1e-16)
        )

    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel, dim=(1,2,3)).view(channels,1,1,1)
    return gaussian_kernel.unsqueeze(1).float()

class get_gaussian_filter(nn.Module):
    def __init__(self, ksize=3, sigma=0, channels=0):
        super(get_gaussian_filter, self).__init__()
        gkernel = get_gaussian_kernel_3d(ksize=ksize, sigma=sigma, channels=channels)
        padding = ksize // 2
        self.gk_layer = nn.Conv3d(in_channels=channels, out_channels=channels,
                                    kernel_size=ksize, groups=channels,
                                    bias=False, padding=padding)
        self.gk_layer.weight.data = gkernel
        self.gk_layer.weight.requires_grad = False
    def forward(self, x):
        return self.gk_layer(x)

#####LoG#####
def get_laplacian_gaussian_kernel_3d(ksize=0, sigma=0, channels=1):
    x_coord = torch.arange(ksize)
    x_grid_2d = x_coord.repeat(ksize).view(ksize, ksize)
    x_grid = x_coord.repeat(ksize*ksize).view(ksize, ksize, ksize)
    y_grid_2d = x_grid_2d.t()
    y_grid  = y_grid_2d.repeat(ksize,1).view(ksize, ksize, ksize)
    z_grid = y_grid_2d.repeat(1,ksize).view(ksize, ksize, ksize)
    xyz_grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).float()
    mean = (ksize - 1)/2.
    variance = sigma**2.
    log_kernel = (-1./(math.pi*(variance**2).view(channels,1,1,1) + 1e-16)) \
                        * (1-(torch.sum((xyz_grid - mean)**2., dim=-1) / (2*variance.view(channels,1,1,1) + 1e-16))) \
                        * torch.exp(-torch.sum((xyz_grid - mean)**2., dim=-1) / (2*variance.view(channels,1,1,1) + 1e-16))
    log_kernel = log_kernel / torch.sum(log_kernel, dim=(1,2,3)).view(channels,1,1,1)
    return log_kernel.unsqueeze(1).float()

class get_laplacian_gaussian_filter(nn.Module):
    def __init__(self, ksize=0, sigma=0, channels=0):
        super(get_laplacian_gaussian_filter, self).__init__()
        gkernel = get_laplacian_gaussian_kernel_3d(ksize=ksize, sigma=sigma, channels=channels)
        padding = ksize // 2
        self.gk_layer = nn.Conv3d(in_channels=channels, out_channels=channels,
                                    kernel_size=ksize, groups=channels,
                                    bias=False, padding=padding)
        self.gk_layer.weight.data = gkernel
        self.gk_layer.weight.requires_grad = False
        
    def forward(self, x):
        return self.gk_layer(x)

####Gabor####
def rotation(theta) : 
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])                            
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])               
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])                                    
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def gabor_fn_3d(sigma=0, thetas=[0,0,0], lambd=3, psi=0, gamma=0.1, ksize=3, channels=1):
    size = ksize // 2
    sigma_x = sigma
    sigma_y = sigma / gamma
    sigma_z = sigma / gamma
    # Bounding box
    (z, y, x) = z, y, x = torch.meshgrid(torch.arange(-size, size + 1), torch.arange(-size, size + 1), torch.arange(-size, size + 1))
    # Rotation
    R = rotation(thetas) 
    z_prime = z * R[0,0] + y * R[0,1] + x * R[0,2]
    y_prime = z * R[1,0] + y * R[1,1] + x * R[1,2]
    x_prime = z * R[2,0] + y * R[2,1] + x * R[2,2]
    gb = torch.exp(-.5 * (x_prime ** 2 / (sigma_x ** 2).view(channels,1,1,1) + 
            y_prime ** 2 / (sigma_y ** 2).view(channels,1,1,1) +
            z_prime ** 2 / (sigma_z ** 2).view(channels,1,1,1))) * torch.cos(2 * math.pi * x_prime / lambd + psi)
    return gb

def get_gabor_kernel_3d(ksize=3, sigma=1, channels=3, theta=0, lambd=3, gamma=0.1, psi=0):
    thetas = [theta, theta, theta]
    gabor_kernel = gabor_fn_3d(sigma=sigma, thetas=thetas, lambd=lambd, psi=psi, gamma=gamma, ksize=ksize, channels=channels)
    gabor_kernel = gabor_kernel.unsqueeze(1).float()
    dummy_kernel = torch.zeros(ksize,ksize)
    dummy_kernel[ksize//2, ksize//2] = 1
    gabor_kernel[sigma==0] = dummy_kernel  
    return gabor_kernel

class get_gabor_filter(nn.Module):
    def __init__(self, ksize=0, sigma=0, channels=0):
        super(get_gabor_filter, self).__init__()
        gkernel = get_gabor_kernel_3d(ksize=ksize, sigma=sigma, channels=channels)
        padding = ksize // 2
        self.gk_layer = nn.Conv3d(in_channels=channels, out_channels=channels,
                            kernel_size=ksize, groups=channels,
                            bias=False, padding=padding)
        self.gk_layer.weight.data = gkernel
        self.gk_layer.weight.requires_grad = False
    def forward(self, x):
        return self.gk_layer(x)


def default_norm_layer(planes, groups=16):
    groups_ = min(groups, planes)
    if planes % groups_ > 0:
        divisor = 16
        while planes % divisor > 0:
            divisor /= 2
        groups_ = int(planes // divisor)
    return nn.GroupNorm(groups_, planes)


def get_norm_layer(norm_type="group"):
    if "group" in norm_type:
        try:
            grp_nb = int(norm_type.replace("group", ""))
            return lambda planes: default_norm_layer(planes, groups=grp_nb)
        except ValueError as e:
            print(e)
            print('using default group number')
            return default_norm_layer
    elif norm_type == "none":
        return None
    else:
        return lambda x: nn.InstanceNorm3d(x, affine=True)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=bias, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, bias=True):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=bias)


class ConvBnRelu(nn.Sequential):

    def __init__(self, inplanes, planes, norm_layer=None, dilation=1, dropout=0):
        if norm_layer is not None:
            super(ConvBnRelu, self).__init__(
                OrderedDict(
                    [
                        ('conv', conv3x3(inplanes, planes, dilation=dilation)),
                        ('bn', norm_layer(planes)),
                        ('relu', nn.ReLU(inplace=True)),
                        ('dropout', nn.Dropout(p=dropout)),
                    ]
                )
            )
        else:
            super(ConvBnRelu, self).__init__(
                OrderedDict(
                    [
                        ('conv', conv3x3(inplanes, planes, dilation=dilation, bias=True)),
                        ('relu', nn.ReLU(inplace=True)),
                        ('dropout', nn.Dropout(p=dropout)),
                    ]
                )
            )


class UBlock(nn.Sequential):
    """Unet mainstream downblock.
    """

    def __init__(self, inplanes, midplanes, outplanes, norm_layer, dilation=(1, 1), dropout=0):
        super(UBlock, self).__init__(
            OrderedDict(
                [
                    ('ConvBnRelu1', ConvBnRelu(inplanes, midplanes, norm_layer, dilation=dilation[0], dropout=dropout)),
                    (
                        'ConvBnRelu2',
                        ConvBnRelu(midplanes, outplanes, norm_layer, dilation=dilation[1], dropout=dropout)),
                ])
        )



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    print(UBlock(4, 4))