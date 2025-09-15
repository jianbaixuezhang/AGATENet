import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CALayer(nn.Module):
    def __init__(self, in_channels=3, reduction=2):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        t = int(math.log2(in_channels)/2+1)
        kernel_size =t if t%2 else t+1
        self.channel_attention = nn.Sequential(
            nn.Conv2d(in_channels, max(1, in_channels // reduction), kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False),
            nn.PReLU(),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, kernel_size=kernel_size,padding=(kernel_size-1)//2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        channel_avg = F.adaptive_avg_pool2d(x, 1)
        channel_weights = self.channel_attention(channel_avg)
        return  channel_weights

class SALayer(nn.Module):
    def __init__(self, in_channels):
        super(SALayer, self).__init__()
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, padding_mode='reflect')
        self.h_conv = nn.Conv2d(in_channels, 1, kernel_size=(1, 3), padding=(0, 1), padding_mode='reflect')
        self.w_conv = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0), padding_mode='reflect')
        self.alpha = nn.Parameter(torch.ones(2))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_feat = torch.mean(x, dim=1, keepdim=True)
        max_feat, _ = torch.max(x, dim=1, keepdim=True)
        spatial_base = self.sigmoid(self.spatial_conv(torch.cat([avg_feat, max_feat], dim=1)))
        h_weights = self.sigmoid(self.h_conv(x))  # 水平方向权重
        w_weights = self.sigmoid(self.w_conv(x))  # 垂直方向权重
        alpha = F.softmax(self.alpha, dim=0)
        spatial_weights = alpha[0] * h_weights + alpha[1] * w_weights
        spatial_weights = spatial_base * spatial_weights
        return  spatial_weights

class PDSCA(nn.Module):

    def __init__(self, in_channels=3, reduction=4):
        super().__init__()
        self.CA=CALayer(in_channels)
        self.SA = SALayer(in_channels)
    def forward(self, x):
        caout = self.CA(x) * x
        saout = self.SA(caout)
        out = caout*saout
        return out
