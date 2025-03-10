# Modified for UniTT-Stereo

# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Masking utils
# --------------------------------------------------------

import random
import torch
import torch.nn as nn
from scipy.stats import truncnorm


class RandomMask(nn.Module):
    """
    random masking
    """

    def __init__(self, num_patches, mask_ratio):
        super().__init__()
        self.num_patches = num_patches
        if type(mask_ratio) != float: #variable ratio
            low = mask_ratio[0]
            up = mask_ratio[1]
            mean = mask_ratio[2]
            std = mask_ratio[3]
            sample = truncnorm.rvs(a = (low-mean)/std, b = (up-mean)/std, loc = mean, scale = std)
            ratio = round(sample,1)
        else: ratio = mask_ratio #fixed ratio
        self.num_mask = int(ratio * self.num_patches)

    def __call__(self, x, mask_ratio=None):
        if mask_ratio is not None:
            self.num_mask = int(mask_ratio * self.num_patches)
        noise = torch.rand(x.size(0), self.num_patches, device=x.device)
        argsort = torch.argsort(noise, dim=1)
        return argsort < self.num_mask
