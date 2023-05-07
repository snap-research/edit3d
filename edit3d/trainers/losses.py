import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch import autograd
import torchvision.utils as vutils

import edit3d

"""
Lap Loss adapted from: https://github.com/mtyka/laploss/blob/master/laploss.py
"""


def gauss_kernel(size=5, sigma=1.0):
    grid = np.float32(np.mgrid[0:size, 0:size].T)
    gaussian = lambda x: np.exp((x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2
    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    return kernel


def conv_gauss(t_input, stride=1, k_size=5, sigma=1.6, repeats=1):
    t_kernel_np = gauss_kernel(size=k_size, sigma=sigma).reshape([1, 1, k_size, k_size])
    t_input_device = edit3d.device
    t_kernel = torch.from_numpy(t_kernel_np).to(t_input_device)
    num_channels = t_input.data.shape[1]
    t_kernel3 = torch.cat([t_kernel] * num_channels, 0)
    t_result = t_input
    for r in range(repeats):
        t_result = F.conv2d(t_result, t_kernel3, stride=1, padding=2, groups=num_channels)
    return t_result


def make_laplacian_pyramid(t_img, max_levels):
    t_pyr = []
    current = t_img
    for level in range(max_levels):
        t_gauss = conv_gauss(current, stride=1, k_size=5, sigma=2.0)
        t_diff = current - t_gauss
        t_pyr.append(t_diff)
        current = F.avg_pool2d(t_gauss, 2, 2)
    t_pyr.append(current)
    return t_pyr


def laploss(t_img1, t_img2, max_levels=3):
    t_pyr1 = make_laplacian_pyramid(t_img1, max_levels)
    t_pyr2 = make_laplacian_pyramid(t_img2, max_levels)
    loss = 0.0
    for i in range(len(t_pyr1)):
        loss += (2 ** (-2 * i)) * L1_loss(t_pyr1[i], t_pyr2[i])
    return loss


def L1_loss(inputs, targets):
    return (inputs - targets).abs().mean()
