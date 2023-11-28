import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import SSIM, CLAHE, calc_uciqe, calc_uiqm
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


class NegMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.Negmetrics = []

    def cast(self, tensor:torch.Tensor):
        tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        image_np = tensor.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        return image_np
    
    def forward(self, input, target):
        x = self.cast(input)
        res = {**calc_uiqm(x), **calc_uciqe(x)}
        for _, lossfunc in enumerate(self.Negmetrics):
            loss = lossfunc(x)
            if isinstance(loss, dict):
                res = {**res, **loss}
            else:
                res[lossfunc._get_name()] = loss
        return res






