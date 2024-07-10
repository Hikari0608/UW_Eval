import torch
import torch.nn as nn
from .NoRef import calc_uciqe, calc_uiqm, calc_niqe
from .Ref import PerScore

class CustomMetric(nn.Module):
    def __init__(self, NoRefMetrics = [
    ]):
        super().__init__()
        self.NoRefMetrics = NoRefMetrics

    def add_metric_per(self, device):
        self.NoRefMetrics.append(PerScore(device=device))

    def cast(self, tensor:torch.Tensor):
        tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
        image_np = tensor.permute(0, 2, 3, 1).squeeze().cpu().numpy()
        return image_np
    
    def forward(self, input, target):
        x = self.cast(input)
        res = {**calc_uiqm(x), **calc_uciqe(x)}
        #self.add_metric_per(input.device)
        for _, lossfunc in enumerate(self.NoRefMetrics):
            loss = lossfunc
            if isinstance(loss, nn.Module):
                res[lossfunc._get_name()] = loss(input, target)
            else:
                res = {**res, **loss(x)}
        return res



