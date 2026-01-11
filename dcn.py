import torch
import torch.nn as nn
import torchvision.ops
from models.common import autopad

class DCN(nn.Module):
    # Deformable Convolution v2
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv_offset = nn.Conv2d(c1, 2 * k * k, kernel_size=k, stride=s, padding=autopad(k, p), bias=True)
        self.conv_mask = nn.Conv2d(c1, k * k, kernel_size=k, stride=s, padding=autopad(k, p), bias=True)
        self.conv = torchvision.ops.DeformConv2d(c1, c2, kernel_size=k, stride=s, padding=autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        # init offset and mask
        self.conv_offset.weight.data.zero_()
        self.conv_offset.bias.data.zero_()
        self.conv_mask.weight.data.zero_()
        self.conv_mask.bias.data.zero_()
        
        # init conv weight
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))
        return self.act(self.bn(self.conv(x, offset, mask)))
