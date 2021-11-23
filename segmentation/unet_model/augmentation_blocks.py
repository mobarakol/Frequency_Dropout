"""The strange way used to perform data augmentation during the Brats 2020 challenge...

Be aware, any batch size above 1 could fail miserably (in an unpredicted way).
"""

from random import randint, random, sample, uniform

import torch
from torch import nn


class DataAugmenter(nn.Module):
    """Performs random flip and rotation batch wise, and reverse it if needed.
    Works"""

    def __init__(self, p=0.5):
        super(DataAugmenter, self).__init__()
        self.p = p
        self.transpose = []
        self.flip = []
        self.toggle = False

    def forward(self, x):
        with torch.no_grad():
            if random() < self.p:
                self.transpose = sample(range(2, x.dim()), 2)
                self.flip = randint(2, x.dim() - 1)
                self.toggle = not self.toggle
                new_x = x.transpose(*self.transpose).flip(self.flip)
                return new_x
            else:
                return x

    def reverse(self, x):
        if self.toggle:
            self.toggle = not self.toggle
            return x.flip(self.flip).transpose(*self.transpose)
        else:
            return x
