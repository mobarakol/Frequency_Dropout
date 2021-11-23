import math

import torch
import torch.nn as nn
import numpy as np

def gabor( ksize=3, sigma=1, theta=torch.tensor(0).float(), lambd=3, gamma=0, psi=0, channels=0):
    xmax, ymax = ksize // 2, ksize // 2
    y, x = torch.meshgrid( torch.arange(-xmax, xmax + 1), torch.arange(-ymax, ymax + 1) )
    x_theta =  x * torch.cos(theta) + y * torch.sin(theta)
    y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
    gauss 	= torch.exp( - ( x_theta**2 + gamma**2 * y_theta **2 ) / (2. * (sigma**2).view(channels,1,1)) )
    grating = torch.cos( (2 * np.pi / lambd) * x_theta + psi )
    return gauss * grating

def get_gabor_kernel_2d(ksize=3, sigma=1, channels=1, theta=torch.tensor(0).float(), lambd = 3, gamma=0):
    gabor_kernel = gabor(ksize=ksize, sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=0, channels=channels)
    gabor_kernel = gabor_kernel.unsqueeze(1).float()
    dummy_kernel = torch.zeros(ksize,ksize)
    dummy_kernel[ksize//2, ksize//2] = 1
    gabor_kernel[sigma==0] = dummy_kernel
    return gabor_kernel

class get_gabor_filter(nn.Module):
    def __init__(self, ksize=3, sigma=0, channels=0, theta=0):
        super(get_gabor_filter, self).__init__()
        gkernel = get_gabor_kernel_2d(ksize=ksize, sigma=sigma, channels=channels, theta=torch.tensor(theta).float())
        padding = ksize // 2
        self.gk_layer = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=ksize, groups=channels,
                                    bias=False, padding=padding)
        self.gk_layer.weight.data = gkernel
        self.gk_layer.weight.requires_grad = False
    def forward(self, x):
        return self.gk_layer(x)

#####Gaussian#####
def get_gaussian_kernel_2d(ksize=0, sigma=0, channels=1):
    x_grid = torch.arange(ksize).repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (ksize - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (1./(2.*math.pi*variance.view(channels,1,1) + 1e-16)) *\
        torch.exp( -torch.sum((xy_grid - mean)**2., dim=-1).view(1, ksize, ksize).repeat(channels,1,1) /\
        (2*variance.view(channels,1,1) + 1e-16)
        )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel, dim=(1,2)).view(channels,1,1)
    return gaussian_kernel.unsqueeze(1).float()

class get_gaussian_filter(nn.Module):
    def __init__(self, ksize=3, sigma=0, channels=0):
        super(get_gaussian_filter, self).__init__()
        sigma = torch.tensor(sigma).repeat(channels) if np.isscalar(sigma) else sigma
        gkernel = get_gaussian_kernel_2d(ksize=ksize, sigma=sigma, channels=channels)

        padding = ksize // 2
        self.gk_layer = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=ksize, groups=channels,
                                    bias=False, padding=padding)
        self.gk_layer.weight.data = gkernel
        self.gk_layer.weight.requires_grad = False
    def forward(self, x):
        return self.gk_layer(x)

#####Laplacian of Gaussian (LoG)#####
def get_laplacian_gaussian_kernel_2d(ksize=0, sigma=0, channels=1):
    x_grid = torch.arange(ksize).repeat(ksize).view(ksize, ksize)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    mean = (ksize - 1)/2.
    variance = sigma**2.
    gaussian_kernel = (-1./(math.pi*(variance**2).view(channels,1,1) + 1e-16)) \
                        * (1-(torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance.view(channels,1,1) + 1e-16))) \
                        * torch.exp(-torch.sum((xy_grid - mean)**2., dim=-1) / (2*variance.view(channels,1,1) + 1e-16))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel, dim=(1,2)).view(channels,1,1)
    return gaussian_kernel.unsqueeze(1).float()

class get_laplacian_gaussian_filter(nn.Module):
    def __init__(self, ksize=0, sigma=0, channels=0):
        super(get_laplacian_gaussian_filter, self).__init__()
        gkernel = get_laplacian_gaussian_kernel_2d(ksize=ksize, sigma=sigma, channels=channels)
        padding = ksize // 2
        self.gk_layer = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=ksize, groups=channels,
                                    bias=False, padding=padding)
        self.gk_layer.weight.data = gkernel
        self.gk_layer.weight.requires_grad = False
        
    def forward(self, x):
        return self.gk_layer(x)
