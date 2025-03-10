# UniTT-Stereo
# For non-commercial purpose only

#----------------------------------
# References
# https://github.com/vinceecws/Monodepth/blob/master/utils/BilinearSampling.py

import torch
import numpy as np
import torch.nn.functional as F

def apply_disparity(img,disp): # gets a warped output
  batch_size, _, height, width = img.size()

  # Original coordinates of pixels
  x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
  y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)

  # Apply shift in X direction
  x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
  flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
  # In grid_sample coordinates are assumed to be between -1 and 1
  output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear', padding_mode='zeros',
  align_corners=True)

  return output