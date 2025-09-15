import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PrecisionDrivenSynergisticContextualAttention import PDSCA
#from TwoDimensionPositionalEncoding import TDimensionPositionalEncoding

# Sparsity enhanced multihead self attention (稀疏增强型多头自注意力)
class SEMHSA(nn.Module):
    def __init__(self, dim, num_heads=4, reduction=4,  bias=False):
        super(SEMHSA, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"

        self.temperature = nn.Parameter(torch.tensor([0.1 * num_heads]), requires_grad=True)   # 这个参数在后续的注意力计算中会起到调节作用
                                                                                             # 比如用于缩放注意力分数等
                                                                                             # 使得注意力机制的计算更具灵活性并且可以通过训练学习到合适的值。
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)  # 变换通道数，生成QKV
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
                                    bias=bias)  # 深度可分离卷积

        #多头注意力的掩码生成
        self.pdcsa_q = PDSCA(in_channels=self.head_dim, reduction=reduction)
        self.pdcsa_k = PDSCA(in_channels=self.head_dim, reduction=reduction)
        #输出投影
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # 后续调整通道数


    def forward(self, x):
        b, c, h, w = x.shape
         #这里的qkv的生成有两种手段，到时候需要对比测试，哪种更好用
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # 重组为多头张量
        def reshape_head(t):
            return rearrange(t, 'b (h d) x y -> (b h) d x y',
                             h=self.num_heads, d=self.head_dim)

        # PDSCA掩码生成（并行处理所有头）
        mask_q = self.pdcsa_q(reshape_head(q))  # [B*H, D, H, W]
        mask_k = self.pdcsa_k(reshape_head(k))

        #多头注意力，初始默认是4，后续调整
        #q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        #v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        # 重组掩码并应用
        mask_q = rearrange(mask_q, '(b h) d x y -> b h (x y) d', b=b, h=self.num_heads)
        mask_k = rearrange(mask_k, '(b h) d x y -> b h (x y) d', b=b, h=self.num_heads)

        # 动态稀疏注意力计算,归一化参数
        q = F.normalize(q * mask_q.sigmoid(), dim=-1)
        k = F.normalize(k * mask_k.sigmoid(), dim=-1)

        # 稀疏注意力矩阵
        attn = (q @ k.transpose(-2, -1)) * self.temperature.exp()  # 指数确保正值
        attn = F.softmax(attn, dim=-1)

        # 值聚合
        out = attn @ v  # [B, H, N, D]

        # 重组输出
        #out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        out = rearrange(out, 'b head (x y) c -> b (head c) x y', x=h, y=w)
        return self.project_out(out)

# 测试代码
if __name__ == "__main__":
    model = SEMHSA(dim=128)
    x = torch.randn(1, 128, 32, 32)
    output = model(x)
    print(output.shape)