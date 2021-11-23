"""A small Unet-like zoo"""
import torch
from torch import nn
from torch.nn import functional as F
from unet_model.layers import (ConvBnRelu, UBlock, conv1x1, get_norm_layer, 
                            get_gaussian_filter, get_laplacian_gaussian_filter, get_gabor_filter)
import random
import numpy as np
#from torchvision.utils import save_image

class Unet(nn.Module):
    """Almost the most basic U-net.
    """
    name = "Unet"

    def __init__(self, inplanes, num_classes, width=48, dropout=0, **kwargs):
        super(Unet, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        print('number of features:',features)
        norm_layer = get_norm_layer('group')
        self.encoder1 = UBlock(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)
        self.encoder2 = UBlock(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)
        self.encoder3 = UBlock(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)
        self.encoder4 = UBlock(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)
        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)
        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)
        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0] // 2, norm_layer, dropout=dropout)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest",)
        self.outconv = conv1x1(features[0] // 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        down1 = self.encoder1(x)
        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)
        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))

        # Decoder
        up3 = self.upsample(bottom_2)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))

        out = self.outconv(up1)

        return out

class Unet_FD(nn.Module):
    """Almost the most basic U-net.
    """
    name = "Unet"

    def __init__(self, inplanes, num_classes, width=48, dropout=0, args=None, **kwargs):
        super(Unet_FD, self).__init__()
        features = [width * 2 ** i for i in range(4)]
        self.features = features

        #CBS hyperparameters
        self.std = args.std
        self.std_factor = args.std_factor
        self.cbs_epoch = args.cbs_epoch
        print('std:', self.std, 'std_factor:', self.std_factor, 'cbs_epoch:', self.cbs_epoch)

        #Frequency Dropout hyperparameters
        print('dropout_p_all:',args.dropout_p_all, '\nfreq_min_all',args.freq_min_all, '\nfreq_max_all',
                args.freq_max_all)
        self.kernel_size = args.kernel_size
        self.filter_all = [get_gaussian_filter, get_laplacian_gaussian_filter, get_gabor_filter]
        self.freq_max_all = args.freq_max_all
        self.freq_min_all = args.freq_min_all
        self.dropout_p_all = args.dropout_p_all
        

        norm_layer = get_norm_layer('group')
        self.encoder1 = UBlock(inplanes, features[0] // 2, features[0], norm_layer, dropout=dropout)
        self.bn1 = nn.BatchNorm3d(features[0])
        self.encoder2 = UBlock(features[0], features[1] // 2, features[1], norm_layer, dropout=dropout)
        self.bn2 = nn.BatchNorm3d(features[1])
        self.encoder3 = UBlock(features[1], features[2] // 2, features[2], norm_layer, dropout=dropout)
        self.bn3 = nn.BatchNorm3d(features[2])
        self.encoder4 = UBlock(features[2], features[3] // 2, features[3], norm_layer, dropout=dropout)
        self.bn4 = nn.BatchNorm3d(features[3])
        self.bottom = UBlock(features[3], features[3], features[3], norm_layer, (2, 2), dropout=dropout)
        self.bottom_2 = ConvBnRelu(features[3] * 2, features[2], norm_layer, dropout=dropout)
        self.downsample = nn.MaxPool3d(2, 2)

        self.decoder3 = UBlock(features[2] * 2, features[2], features[1], norm_layer, dropout=dropout)
        self.decoder2 = UBlock(features[1] * 2, features[1], features[0], norm_layer, dropout=dropout)
        self.decoder1 = UBlock(features[0] * 2, features[0], features[0] // 2, norm_layer, dropout=dropout)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest",)
        self.outconv = conv1x1(features[0] // 2, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def random_selection(self, in_ch, f_idx):       
        dropout_p =  self.dropout_p_all[f_idx]
        freq_min, freq_max = self.freq_min_all[f_idx], self.freq_max_all[f_idx]
        sigma = torch.tensor(np.random.uniform(freq_min,freq_max, in_ch))
        dropout = nn.Dropout(dropout_p)
        sigma = dropout(sigma)*(1-dropout_p)
        return sigma

    def get_new_kernels(self):
        f_idx = random.randint(0,2)
        filter_func = self.filter_all[f_idx]
        sigma = self.random_selection(self.features[0],f_idx)
        self.kernel1 = filter_func(ksize=self.kernel_size, sigma=sigma, channels=self.features[0],)
        sigma = self.random_selection(self.features[1],f_idx)
        self.kernel2 = filter_func(ksize=self.kernel_size, sigma=sigma, channels=self.features[1],)
        sigma = self.random_selection(self.features[2],f_idx)
        self.kernel3 = filter_func(ksize=self.kernel_size, sigma=sigma, channels=self.features[2],)
        sigma = self.random_selection(self.features[3],f_idx)
        self.kernel4 = filter_func(ksize=self.kernel_size, sigma=sigma, channels=self.features[3],)

    def get_new_kernels_cbs(self, epoch_count):
        if epoch_count % self.cbs_epoch == 0 and epoch_count is not 0:
            self.std *= self.std_factor
        self.kernel1 = get_gaussian_filter(ksize=self.kernel_size, sigma=torch.tensor(self.std).repeat(self.features[0]), channels=self.features[0])
        self.kernel2 = get_gaussian_filter(ksize=self.kernel_size, sigma=torch.tensor(self.std).repeat(self.features[1]), channels=self.features[1])
        self.kernel3 = get_gaussian_filter(ksize=self.kernel_size, sigma=torch.tensor(self.std).repeat(self.features[2]), channels=self.features[2])
        self.kernel4 = get_gaussian_filter(ksize=self.kernel_size, sigma=torch.tensor(self.std).repeat(self.features[3]), channels=self.features[3])

    def forward(self, x):
        down1 = self.encoder1(x)
        down1 = F.relu(self.bn1(self.kernel1(down1)))
        down2 = self.downsample(down1)
        down2 = self.encoder2(down2)
        down2 = F.relu(self.bn2(self.kernel2(down2)))
        down3 = self.downsample(down2)
        down3 = self.encoder3(down3)
        down3 = F.relu(self.bn3(self.kernel3(down3)))
        down4 = self.downsample(down3)
        down4 = self.encoder4(down4)
        down4 = F.relu(self.bn4(self.kernel4(down4)))
        bottom = self.bottom(down4)
        bottom_2 = self.bottom_2(torch.cat([down4, bottom], dim=1))

        # Decoder
        up3 = self.upsample(bottom_2)
        up3 = self.decoder3(torch.cat([down3, up3], dim=1))
        up2 = self.upsample(up3)
        up2 = self.decoder2(torch.cat([down2, up2], dim=1))
        up1 = self.upsample(up2)
        up1 = self.decoder1(torch.cat([down1, up1], dim=1))

        out = self.outconv(up1)

        return out
