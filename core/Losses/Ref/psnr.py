import torch
import torch.nn as nn
import torch.nn.functional as F


class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()

    def forward(self, x, y):
        mse = F.mse_loss(x, y)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr