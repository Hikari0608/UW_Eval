from torch import Tensor
from .Ref import SSIM

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import L1Loss as L1, MSELoss as MSE
import numpy as np
from .base import CLAHE

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

class baseLoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(baseLoss, self).__init__(size_average, reduce, reduction)
        self.losses = []
        self.reduction = reduction

    def add(self, func):
        if isinstance(func, list):
            self.losses.extend(func)
        else:
            self.losses.append(func)

    def forward(self, input, target):
        pass
        
    def calculate(self, input, target) -> dict:
        res = {}
        for _, lossfunc in enumerate(self.losses):
            loss = lossfunc(input, target)
            if isinstance(loss, dict):
                res = {**res, **loss}
            else:
                res[lossfunc._get_name()] = loss
        return res


class AddLoss(baseLoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(AddLoss, self).__init__(size_average, reduce, reduction)
    
    def forward(self, input, target) -> Tensor:
        losses = list(self.calculate(input, target).values())
        loss = losses[0]
        for _ in losses:
            loss += _
        return loss

class MulLoss(AddLoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean'):
        super(MulLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        losses = list(self.calculate(input, target).values())
        loss = losses[0]
        for _ in losses:
            loss *= _
        return loss
    

class NoRefLoss(AddLoss):
    def __init__(self, func):
        super(MulLoss, self).__init__()
        self.func = func

    def forward(self, input, target):
        return self.func(input)


class MSELoss(MSE):
    def __init__(self, loss_weight=1.0, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.loss_weight = loss_weight

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.loss_weight * super().forward(input, target)

class MAELoss(L1):
    def __init__(self, loss_weight=1.0, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.loss_weight = loss_weight
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.loss_weight * super().forward(input, target)

class SSIMLoss(SSIM):
    def __init__(self, window_size=11, size_average=True, loss_weight=1.0):
        super(SSIMLoss, self).__init__(window_size=window_size, size_average=size_average)
        self.loss_weight = loss_weight

    def forward(self, img1, img2):
        loss = self.loss_weight * (1 - super().forward(img1, img2))
        return loss


class CLAHELoss(MAELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', loss_weight=1.0) -> None:
        super().__init__(size_average, reduce, reduction)
        self.clahe = CLAHE()
        self.loss_weight = loss_weight
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input_tensor = input * 255
        input_array = input_tensor.permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)
        out_array = self.clahe(input_array)
        out_tensor = torch.tensor(out_array, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
        out = out_tensor.to(input.device)
        return self.loss_weight * super().forward(input, out)

