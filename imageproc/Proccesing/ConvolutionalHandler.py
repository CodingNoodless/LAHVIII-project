import math

import numpy as np
from torch import nn
import cv2
import tarfile
import os
import glob
from PIL import Image


# lets first explain some logic:
#     - we want to upsacle a low res image to a higher res image
#     - we dont want to lose geometric dataclasses
#     - we want to make it fast ENOUGHT
#
# so...
#  - we use resnet blocks (conv -> relu -> conv)


class ResSequential(nn.Module):
    def __init__(self, layers, res_scale):
        super(ResSequential, self).__init__()
        self.res_scale = res_scale
        self.layers = nn.Sequential(*layers)

    def forward(self, x): return x + self.layers * self.res_scale


def conv(features, inputs, kernel_size=3, atEndofBlock=False):
    layers = [nn.Conv2d(inputs, features, kernel_size, padding=kernel_size // 2)]
    if atEndofBlock:
        layers.append(nn.ReLU(True))
    nn.Sequential(*layers)
    # resnet blocks are conv->relu->conv, this func sets it up easily


def res_block(ni): return [conv(ni, ni), conv(ni, ni, atEndofBlock=True)]


# basic resnet block


def upsample(inputs, features, scale=10):
    layers = []
    for i in range(int(math.log(scale, 2))):
        layers += [conv(inputs, features * 4), nn.PixelShuffle(2)]
    return nn.Sequential(*layers)


# upsampled block

class deCNN(nn.Module):
    def __init__(self, scale, features=64, hidden_layers=8):
        super(deCNN, self).__init__()
        layers = []
        # initial conv layer, 3 channels for H, S, V

        layers.append(conv(3, features))
        for i in range(hidden_layers):
            layers.append(conv(features, features))
            layers.append(upsample(features, features, scale))
            layers.append(nn.BatchNorm2d(features))
            layers.append(conv(features, 3, atEndofBlock=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, input): return self.layers(input)
