import math
import numpy as np
from torch import nn


class PSNR(nn.Module):
    def __init__(self): super(PSNR, self)


class Adversarial(nn.Module):
    def __init__(self): super(Adversarial, self)


class Standard(nn.Module):
    def __init__(self):
        super(Standard, self).__init__()
        self.Learner = Learner()
