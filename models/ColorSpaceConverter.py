import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBLABConverter(nn.Module):

    def __init__(self):
        super().__init__()

        # CIE标准参数
        self.register_buffer('rgb2xyz_mat', torch.tensor([
            [0.412453, 0.357580, 0.180423],
            [0.212671, 0.715160, 0.072169],
            [0.019334, 0.119193, 0.950227]
        ]))
        self.register_buffer('xyz2rgb_mat', torch.tensor([
            [3.24048134, -1.53715152, -0.49853633],
            [-0.96925495, 1.87599, 0.04155593],
            [0.05564664, -0.20404134, 1.05731107]
        ]))
        self.register_buffer('ref_white', torch.tensor([0.95047, 1.0, 1.08883]))
        self.register_buffer('epsilon', torch.tensor(216 / 24389.0))  # 6^3/29^3
        self.register_buffer('kappa', torch.tensor(24389 / 27))  # 29^3/3^3
        self.register_buffer('eps', torch.tensor(1e-6))

    def _normalize_lab(self, l, a, b):
        """LAB全通道归一化到[-1,1]"""
        l = (l / 50.0) - 1.0  # 原始范围[0,100] → [-1,1]
        a = a / 128.0  # 原始范围[-128,127] → [-1,~0.992]
        b = b / 128.0
        return l, a, b

    def _denormalize_lab(self, l, a, b):
        """从[-1,1]反归一化到原始范围"""
        l = (l + 1.0) * 50.0  # [-1,1] → [0,100]
        a = a * 128.0  # [-1,1] → [-128,128]
        b = b * 128.0
        return l, a, b

    def rgb_to_lab(self, rgb):
        """RGB转归一化LAB"""
        # Gamma校正
        rgb = torch.where(rgb > 0.04045,
                          ((rgb + 0.055) / 1.055) ** 2.4,
                          rgb / 12.92)

        # RGB→XYZ
        xyz = torch.einsum('ij,bjhw->bihw', self.rgb2xyz_mat, rgb)
        xyz = xyz / self.ref_white.view(1, 3, 1, 1)  # 白点归一化

        # XYZ→LAB
        mask = xyz > self.epsilon
        f_xyz = torch.where(mask, xyz ** (1 / 3), (self.kappa * xyz + 16) / 116)

        l = 116 * f_xyz[:, 1:2] - 16
        a = 500 * (f_xyz[:, 0:1] - f_xyz[:, 1:2])
        b = 200 * (f_xyz[:, 1:2] - f_xyz[:, 2:3])

        # 全通道归一化
        l, a, b = self._normalize_lab(l, a, b)
        return torch.cat([l, a, b], dim=1).clamp(-1.0, 1.0)

    def lab_to_rgb(self, lab):
        """归一化LAB转RGB"""
        # 反归一化
        l, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
        l, a, b = self._denormalize_lab(l, a, b)

        # 保证物理有效性
        l = l.clamp(0, 100)
        a = a.clamp(-128, 128)
        b = b.clamp(-128, 128)

        # LAB→XYZ
        y = (l + 16) / 116
        x = a / 500 + y
        z = y - b / 200

        xyz = torch.cat([x, y, z], dim=1)
        mask = xyz > self.epsilon
        xyz = torch.where(mask, xyz ** 3, (116 * xyz - 16) / self.kappa)
        xyz = xyz * self.ref_white.view(1, 3, 1, 1)

        # XYZ→RGB
        rgb = torch.einsum('ij,bjhw->bihw', self.xyz2rgb_mat, xyz)
        rgb = torch.where(rgb > 0.0031308,
                          1.055 * (rgb ** (1 / 2.4)) - 0.055,
                          12.92 * rgb)
        return torch.clamp(rgb, 0.0, 1.0)

    def forward(self, x, mode):
        if mode == 'rgb2lab':
            return self.rgb_to_lab(x)
        elif mode == 'lab2rgb':
            return self.lab_to_rgb(x)
        else:
            raise ValueError(f"Invalid mode: {mode}")

