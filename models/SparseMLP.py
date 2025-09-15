import torch
import torch.nn as nn


#sparse mlp (稀疏MLP)
class SparseMLP(nn.Module):
    def __init__(self, W, H, channels):
        super().__init__()
        assert W == H
        self.channels = channels
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(channels)
        self.proj_h = nn.Conv2d(H, H, (1, 1))
        self.proh_w = nn.Conv2d(W, W, (1, 1))
        self.fuse = nn.Conv2d(channels*3, channels, (1,1), (1,1), bias=False)

    def forward(self, x):
        x = self.activation(self.BN(x))
        x_h = self.proj_h(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_w = self.proh_w(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x = self.fuse(torch.cat([x, x_h, x_w], dim=1))
        return x


#稀疏连接，权值共享
#只与同一行或同列上的token直接交互，而不是与所有其他topken交互。此外，所有行和所有列可以分别共享相同的投影权重。
#这样的方法，可以实现全局的感受野
