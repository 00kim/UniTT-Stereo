# UniTT-Stereo
# For non-commercial purpose only

#----------------------------------
# References
# https://github.com/vinceecws/Monodepth/blob/master/Loss.py
# https://github.com/jiaw-z/FCStereo/blob/f76c3317e0951986b49a3bb794028a8ae067d410/dmb/modeling/stereo/losses/contrastive_loss.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils import BilinearSampling as bs

class ConsistencyLoss(nn.Module):

    # alpha_AP = Appearance Matching Loss weight
    # alpha_DS = Disparity Smoothness Loss weight
    # alpha_LR = Left-Right Consistency Loss weight

    def __init__(self, n=4, alpha_AP=0.85, alpha_DS=0.1, alpha_LR=1.0):
        super(ConsistencyLoss, self).__init__()

        self.n = n
        self.alpha_AP = alpha_AP
        self.alpha_DS = alpha_DS
        self.alpha_LR = alpha_LR

    def get_images(self, pyramid, disp, get):
        if get == 'left':
            return bs.apply_disparity(pyramid, -disp)
        elif get == 'right':
            return [bs.apply_disparity(pyramid[j], disp[j]) for j in range(self.n)]
        else:
            raise ValueError('Argument get must be either \'left\' or \'right\'')


    def L1(self, pyramid, est, mask, valid):

        L1_loss = torch.mean(torch.abs((pyramid - est) * valid))

        return L1_loss

    def get_loss(self, left_pyramid, left_est, mask, valid):

        # L1 Loss
        left_l1 = self.L1(left_pyramid, left_est, mask, valid)

        return left_l1

    def forward(self, disp, target, epoch, args, imgs, patch, mask=None):

        h = imgs.shape[2] // patch
        w = imgs.shape[3] // patch

        left, right = target
        left = left.reshape(1, left.shape[2], h, w*args.batch_size).cuda()
        right = right.reshape(1, right.shape[2], h, w*args.batch_size).cuda()

        gtcopy = disp.clone().cuda()
        valid = torch.isfinite(gtcopy)
        gtcopy[~valid] = 0

        disps_lowr = F.interpolate(gtcopy, scale_factor=(1/16), mode='nearest').reshape(1, 1, h, w*args.batch_size) /16

        valid = (disps_lowr > 0)
        mask = mask.reshape(1,h,w*args.batch_size)

        # Reconstructed images
        left_est = self.get_images(right, disps_lowr, 'left').cuda()

        # AP Loss
        AP_loss = self.get_loss(left, left_est, mask, valid)

        return AP_loss



