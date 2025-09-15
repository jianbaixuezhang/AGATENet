import torch
import torch.nn as nn
import torch.nn.functional as F


class TDimensionPositionalEncoding(nn.Module):
    def __init__(self, dim, max_h=128, max_w=128):
        super().__init__()

        # 创建网格坐标（-1 到 1）二维空间中的位置信息
        y_embed = torch.linspace(-1., 1., steps=max_h)
        x_embed = torch.linspace(-1., 1., steps=max_w)
        y_grid, x_grid = torch.meshgrid(y_embed, x_embed, indexing='ij')
        grid = torch.stack([x_grid, y_grid], dim=0)  # [2, H, W]
        # 在初始化时，创建位置编码
        pe = self._generate_pe(grid, dim)
        # 位置编码通过 register_buffer 保持在 GPU 或 CPU
        self.register_buffer('pe', pe)
        # 学习的尺度因子，用于调整位置编码对最终结果的影响程度
        self.scale = nn.Parameter(torch.tensor(0.1))

    def _generate_pe(self, grid, dim):
        # 将网格坐标转换为 [1, 2, H, W] 形式
        grid = grid.unsqueeze(0)  # [1, 2, H, W]
        # 使用卷积生成位置编码
        # [1, 2, H, W] -> [1, dim, H, W]
        pe = nn.Conv2d(2, dim, 1, groups=2)(grid)
        return pe

    def forward(self, x):
        b, c, h, w = x.shape
        pe = self.pe  # 从缓冲区获取位置编码
        if pe.size(2) != h or pe.size(3) != w:
            # 如果位置编码的尺寸和输入不匹配，进行插值调整
            pe = F.interpolate(pe, size=(h, w), mode='bilinear', align_corners=True)
        # 扩展位置编码以匹配batch维度
        pe = pe.expand(b, -1, -1, -1)
        # 返回加了位置编码后的特征图
        return x + self.scale * pe


class RelativePositionEncoder(nn.Module):
    def __init__(self, dim=2, window_size=7):
        super().__init__()
        self.window_size = window_size

        # 生成归一化的绝对坐标网格（与TDimensionPositionalEncoding一致）
        coords_h = torch.linspace(-1., 1., steps=window_size)
        coords_w = torch.linspace(-1., 1., steps=window_size)
        y_grid, x_grid = torch.meshgrid(coords_h, coords_w, indexing='ij')
        grid = torch.stack([x_grid, y_grid], dim=0)  # [2, Wh, Ww]
        self.register_buffer("grid", grid.unsqueeze(0))  # [1, 2, Wh, Ww]

        # 使用1x1卷积生成位置编码（输入2通道，输出dim通道）
        self.conv = nn.Conv2d(2, dim, kernel_size=1)
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self):
        # 输入形状: [1, 2, Wh, Ww]
        # 通过卷积生成位置编码: [1, dim, Wh, Ww]
        pe = self.conv(self.grid)

        # 展开为序列形式: [1, dim, Wh*Ww]
        pe_flatten = pe.view(1, pe.size(1), -1)  # [1, dim, Wh*Ww]

        # 计算相对位置编码（差值）: [1, Wh*Ww, Wh*Ww, dim]
        relative_pe = pe_flatten.permute(0, 2, 1)[:, :, None, :] - \
                      pe_flatten.permute(0, 2, 1)[:, None, :, :]

        # 调整维度: [Wh*Ww, Wh*Ww, dim]
        relative_pe = relative_pe.squeeze(0)  # [Wh*Ww, Wh*Ww, dim]

        return self.scale * relative_pe