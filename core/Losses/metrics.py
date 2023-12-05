import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .losses import baseLoss


class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, x, y):
        mse = F.mse_loss(x, y)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr


class MSE(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x ,y):
        return F.mse_loss(x, y)


class Metrics(baseLoss):
    def __init__(self):
        super(Metrics, self).__init__()
        self.clear()

class Metrics(baseLoss):
    def __init__(self):
        super(Metrics, self).__init__()
        self.clear()

    def clear(self):
        self.metric = {}
        self.tag = False
        self.last = {}

    def output(self, num_data) -> dict:
        for key, tensor in self.metric.items():
            self.metric[key] /= num_data
        return self.metric
    
    def back(self) -> dict:
        return self.last

    def forward(self, input, target) -> None:
        res = super().calculate(input, target)
        self.last = res
        if not(self.tag):
            self.metric = res
            self.tag = True
        else:
            for key, tensor in res.items():
                self.metric[key] += tensor
        return res









