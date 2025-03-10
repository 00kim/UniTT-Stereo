# Modified for UniTT-Stereo

# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# UniTT-Stereo
# --------------------------------------------------------
import torch
import torchvision
from torch import nn
import random

from .blocks import Block
from .croco import CroCoNet
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from scipy.stats import truncnorm

def croco_args_from_ckpt(ckpt):
    if 'croco_kwargs' in ckpt:  # CroCo v2 released models
        return ckpt['croco_kwargs']
    elif 'args' in ckpt and hasattr(ckpt['args'], 'model'):  # pretrained using the official code release
        s = ckpt['args'].model  # eg "CroCoNet(enc_embed_dim=1024, enc_num_heads=16, enc_depth=24)"
        assert s.startswith('CroCoNet(')
        return eval('dict' + s[len('CroCoNet'):])  # transform it into the string of a dictionary and evaluate it
    else:  # CroCo v1 released models
        return dict()

class UniTTStereo(CroCoNet):

    def __init__(self,
                 head,
                 **kwargs):
        """ Build network for binocular downstream task
        It takes an extra argument head, that is called with the features
          and a dictionary img_info containing 'width' and 'height' keys
        The head is setup with the croconet arguments in this init function
        """
        super(UniTTStereo, self).__init__(**kwargs)
        head.setup(self)
        self.head = head

    def append_mask(self, feat1, masks1):
        # append masked tokens to the sequence
        B, Nenc, C = feat1.size()
        if masks1 is None:  # downstreams
            f1_ = feat1
        else:  # pretraining
            Ntotal = masks1.size(1)
            func = nn.Parameter(torch.zeros(1, 1, self.enc_embed_dim))
            f1_ = func.repeat(B, Ntotal, 1).to(dtype=feat1.dtype).cuda()
            # apply Transformer blocks
            f1_[~masks1] = feat1.view(B * Nenc, C)
        return f1_

    def _encode_image(self, image, do_mask=False, return_all_blocks=False, epoch=None, args=None):
        """
        image has B x 3 x img_size x img_size
        do_mask: whether to perform masking or not
        return_all_blocks: if True, return the features at the end of every block
                           instead of just the features from the last block (eg for some prediction heads)
        """
        # embed the image into patches  (x has size B x Npatches x C)
        # and get position if each return patch (pos has size B x Npatches x 2)
        x, pos = self.patch_embed(image)
        # add positional embedding without cls token
        if self.enc_pos_embed is not None:
            x = x + self.enc_pos_embed[None, ...]
        # apply masking
        B, N, C = x.size()
        if do_mask:
            ratio = args.mask_ratio
            low = ratio[0]
            up = ratio[1]
            mean = ratio[2]
            std = ratio[3]
            sample = truncnorm.rvs(a = (low-mean)/std, b = (up-mean)/std, loc = mean, scale = std)
            random_ratio = round(sample,1)
            masks = self.mask_generator(x, mask_ratio=random_ratio)
            x = x[~masks].view(B, -1, C)
            posvis = pos[~masks].view(B, -1, 2)
        else:
            B, N, C = x.size()
            masks = torch.zeros((B, N), dtype=bool)
            posvis = pos
        attn_scores = []
        #cls_token = x.mean(dim=1).unsqueeze(1)
        # now apply the transformer encoder and normalization
        #x = torch.cat((cls_token, x), dim=1)
        if return_all_blocks:
            out = []
            num = 1
            for blk in self.enc_blocks:
                x, attn_score = blk(x, posvis, num)
                num +=1
                out.append(x)
                attn_scores.append(attn_score)
            out[-1] = self.enc_norm(out[-1])
            if do_mask:
                return out, pos, masks, random_ratio
            return out, pos, masks, attn_scores
        else:
            num=1
            for blk in self.enc_blocks:
                x, attn_score = blk(x, posvis, num)
                num+=1
                attn_scores.append(attn_score)
            x = self.enc_norm(x)
            if do_mask:
                return x, pos, masks, random_ratio
            return x, pos, masks, attn_scores

    def forward(self, img1, img2, epoch=None, args=None, train=None):
        B, C, H, W = img1.size()
        img_info = {'height': H, 'width': W}
        return_all_blocks = hasattr(self.head, 'return_all_blocks') and self.head.return_all_blocks

        if train is not None:  # train: do_mask=True
            out, pos, mask1, ratio = self._encode_image(img1, do_mask=True, return_all_blocks=return_all_blocks,
                                                        epoch=epoch, args=args)
            out2, pos2, _, _ = self._encode_image(img2, do_mask=False, return_all_blocks=False, epoch=epoch)
            if return_all_blocks:
                decout, attn_scores, cross_attn_scores = self._decoder(out[-1], pos, mask1, out2, pos2, return_all_blocks=return_all_blocks)
                recon= self.prediction_head(decout[decout.__len__() - 1])
                decout = out + decout
            else:
                decout, attn_scores, cross_attn_scores = self._decoder(out, pos, mask1, out2, pos2, return_all_blocks=return_all_blocks)
                recon = self.prediction_head(decout)
            target = self.patchify(img1)

            out = out[out.__len__() - 1]

            feat1 = self.append_mask(out, mask1)
           

            return self.head(decout, img_info), recon, mask1, target, feat1, out2, ratio, decout

        else:  # validate or test
            out, pos, _, enc_attn_scores = self._encode_image(img1, do_mask=False, return_all_blocks=return_all_blocks)
            out2, pos2, _, right_attn_scores = self._encode_image(img2, do_mask=False, return_all_blocks=False)

            if return_all_blocks:
                decout, dec_attn_scores, cross_attn_scores = self._decoder(out[-1], pos, None, out2, pos2, return_all_blocks=return_all_blocks)
                decout = out + decout
            else:
                decout, dec_attn_scores, cross_attn_scores = self._decoder(out, pos, None, out2, pos2, return_all_blocks=return_all_blocks)

            return self.head(decout, img_info), decout, enc_attn_scores, dec_attn_scores, cross_attn_scores