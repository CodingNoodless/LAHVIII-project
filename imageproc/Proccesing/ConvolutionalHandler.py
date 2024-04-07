import math

import numpy as np
from torch import nn
import torch
import cv2
import tarfile
import os
import glob
from PIL import Image
import torch.nn.functional as F


# lets first explain some logic:
#     - we want to upsacle a low res image to a higher res image
#     - we dont want to lose geometric dataclasses
#     - we want to make it fast ENOUGHT
#
# so...
#  - we use resnet blocks (conv -> relu -> conv)


class ResSequential(nn.Module):
    def __init__(self, layers, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.layers = nn.Sequential(*layers)

    def forward(self, x): return x + self.layers(x) * self.res_scale


def conv(features, inputs, kernel_size=3, atcn=True):
    layers = [nn.Conv2d(inputs, features, kernel_size, padding=kernel_size // 2)]
    if atcn:
        layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)
    # resnet blocks are conv->relu->conv, this func sets it up easily


def res_block(ni): return ResSequential([conv(ni, ni), conv(ni, ni, False)], 0.1)


# basic resnet block


def upsample(inputs, features, scale=10):
    layers = []
    for i in range(int(math.log(scale, 2))):
        layers += [conv(inputs, features * 4), nn.PixelShuffle(2)]
    return nn.Sequential(*layers)


# upsampled block

class deCNN(nn.Module):
    def __init__(self, scale, features=3, hidden_layers=8):
        super().__init__()
        layers = []
        # initial conv layer, 3 channels for H, S, V
        self.scale = scale

        layers.append(conv(3, features))
        for i in range(hidden_layers): layers.append(res_block(features))
        layers.append(conv(features, features))
        layers.append(upsample(features, features, scale))
        layers.append(nn.BatchNorm2d(features))
        layers.append(conv(features, 3, False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return x

class SuperResolutionLoss(nn.Module):
    def __init__(self):
        super(SuperResolutionLoss, self).__init__()

    def forward(self, output, target):
        # output = output.view(-1)
        # target = target.view(-1)
        loss = F.mse_loss(output, target)
        return loss
