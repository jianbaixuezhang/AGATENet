import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import math
from  models.PrecisionDrivenSynergisticContextualAttention import PDSCA
from  models.ColorSpaceConverter import RGBLABConverter
#from models.deform import DeformConv2d
#稀疏连接，权值共享，稀疏的mlp
#考虑空间信息
class SparseMLP(nn.Module):
	def __init__(self,network_depth, channels,hidden_channels = None, out_channels = None):
		super().__init__()
		self.channels = channels
		hidden_channels = hidden_channels or channels  # 新增默认值
		out_channels = out_channels or channels  # 新增默认值
		assert hidden_channels >= channels, (
			f"hidden_channels ({hidden_channels}) should >= channels ({channels})"
		)
		self.network_depth = network_depth
		self.activation = nn.PReLU()
		self.BN = nn.BatchNorm2d(channels)
		# 空间投影
		self.proj_h = nn.Conv1d(channels, channels, 1)  # 高度方向投影
		self.proj_w = nn.Conv1d(channels, channels, 1)  # 宽度方向投影
		# 融合层
		self.fuse = nn.Sequential(
            nn.Conv2d(3*channels, hidden_channels, 1),
			nn.PReLU(),
            nn.Conv2d(hidden_channels, out_channels, 1)
        )

	def forward(self, x):
		B, C, H, W = x.shape  # 动态获取输入尺寸
		# 基础处理
		x = self.activation(self.BN(x))

		# 高度方向投影
		x_h = x.permute(0, 3, 1, 2)  # [B, W, C, H]
		x_h = x_h.reshape(B * W, C, H)  # [B*W, C, H]
		x_h = self.proj_h(x_h)  # [B*W, C, H]
		x_h = x_h.reshape(B, W, C, H)  # [B, W, C, H]
		x_h = x_h.permute(0, 2, 3, 1)  # [B, C, H, W]

		# 宽度方向投影
		x_w = x.permute(0, 2, 1, 3)  # [B, H, C, W]
		x_w = x_w.reshape(B * H, C, W)  # [B*H, C, W]
		x_w = self.proj_w(x_w)  # [B*H, C, W]
		x_w = x_w.reshape(B, H, C, W)  # [B, H, C, W]
		x_w = x_w.permute(0, 2, 1, 3)  # [B, C, H, W]

		# 特征融合
		fused = torch.cat([x, x_h, x_w], dim=1)  # [B, 3C, H, W]
		return self.fuse(fused)  # [B, C, H, W]


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 参数保持兼容性（kernel_size参数可忽略）
        self.proj = nn.Sequential(
            nn.PixelUnshuffle(patch_size),  # 空间维度压缩
            nn.Conv2d(in_chans * (patch_size**2), embed_dim, kernel_size=1)  # 通道调整
        )

    def forward(self, x):
        return self.proj(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1  # 默认使用1x1卷积

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size**2,
                      kernel_size=kernel_size,
                      padding=kernel_size//2,
                      padding_mode='reflect',
                      bias=False),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        return self.proj(x)
#划分窗口
def window_partition(x, window_size):
	B, H, W, C = x.shape
	x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
	windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, C) # B*H//window_size*W//window_size, window_size**2, C
																				   #每个小窗口内的元素在维度上排列在一起，方便后续的处理
	return windows

#窗口恢复
def window_reverse(windows, window_size, H, W):
	B = int(windows.shape[0] / (H * W / window_size / window_size))
	x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
	x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
	return x

#计算指定窗口下的位置信息
class get_TD_positions(nn.Module):
	def __init__(self, dim=2, window_size=7):
		super().__init__()
		self.window_size = window_size
		# 生成归一化的绝对坐标网格
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


#聚焦输入的窗口内的注意力计算，注意输入的形参
class WindowAttention(nn.Module):
	def __init__(self, dim, window_size, num_heads, reduction=4,  bias=False):
		#在多头注意力机制中，通过设置多个头可以让模型从不同的角度去关注特征，并行地捕捉多种特征关联关系，最后再进行整合。
		super().__init__()
		self.dim = dim
		self.window_size = window_size  # Wh, Ww
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"
		self.log_temp = nn.Parameter(torch.tensor([math.log(0.1)]))  # 初始温度=0.1*num_heads

		self.register_buffer('num_heads_tensor', torch.tensor(num_heads, dtype=torch.float32))
		self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)  # 变换通道数，生成QKV
		self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
									bias=bias)  # 深度可分离卷积

		# 多头注意力的掩码生成
		self.pdcsa_q = PDSCA(in_channels=self.head_dim, reduction=reduction)
		self.pdcsa_k = PDSCA(in_channels=self.head_dim, reduction=reduction)
		# 输出投影
		self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)  # 后续调整通道数
		#获取窗口内的相对位置信息
		self.positions = get_TD_positions(dim=self.dim, window_size=self.window_size)

		self.meta = nn.Sequential(
			nn.Linear(self.dim, 128),
			nn.GELU(),
			nn.Linear(128,self.num_heads)  #将提取和处理后的相对位置特征对应到每个注意力头
															# 便于后续为每个注意力头添加相应的相对位置偏置信息。
		)												  #[Wh * Ww, Wh * Ww, num_heads]

		self.softmax = nn.Softmax(dim=-1)


	def forward(self, qkv):
		#b, c, h, w = x.shape
		B_, N, _ = qkv.shape
		qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)  # 形状重塑和维度重排
		# 查询（q）、键（k）、值（v）这 3 个部分
		q, k, v = qkv[0], qkv[1], qkv[2]
		temperature = torch.exp(self.log_temp)

		# PDSCA掩码生成（并行处理所有头）
		mask_q = self.pdcsa_q(q.reshape(-1, self.head_dim, self.window_size, self.window_size))
		mask_k = self.pdcsa_k(k.reshape(-1, self.head_dim, self.window_size, self.window_size))
		q = q * mask_q.view_as(q).sigmoid()  # 仅用掩码加权，不归一化
		k = k * mask_k.view_as(k).sigmoid()

		attn = (q @ k.transpose(-2, -1)) * temperature# 指数确保正值
		positions = self.positions().to(self.log_temp.device)
		position_bias = self.meta(positions).permute(2, 0, 1).contiguous()
		attn = attn + position_bias.unsqueeze(0)
		attn = self.softmax(attn)
		out = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim) # [B, H, N, D]
		out = self.project_out(out.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)

		return out

# Sparsity enhanced multihead self attention (稀疏增强型多头自注意力)
class SEMHSA(nn.Module):
	def __init__(self, network_depth, dim, num_heads, window_size, shift_size):
		super().__init__()
		self.dim = dim
		self.head_dim = int(dim // num_heads)
		self.num_heads = num_heads

		self.window_size = window_size
		self.shift_size = shift_size		# 窗口的滑动步长大小，用于实现SW-MSA
		self.network_depth = network_depth  # 网络总深度


		self.QK = nn.Conv2d(dim, dim * 2, 1)
		self.V = nn.Conv2d(dim, dim, 1)
		self.attn = WindowAttention(dim, window_size, num_heads)
		self.proj = nn.Conv2d(dim, dim, 1)  # 输出投影层
	#检查与形状填充
	def check_size(self, x, shift=False):
		_, _, h, w = x.size()
		mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
		mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

		if shift:
			x = F.pad(x, (self.shift_size, (self.window_size - self.shift_size + mod_pad_w) % self.window_size,
						  self.shift_size, (self.window_size - self.shift_size + mod_pad_h) % self.window_size),
					  mode='reflect')
		else:
			x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
		return x

	def forward(self, X):
		B, C, H, W = X.shape
		QK = self.QK(X)
		V = self.V(X)
		QKV = torch.cat([QK, V], dim=1)
		# shift
		shifted_QKV = self.check_size(QKV, self.shift_size > 0)
		Ht, Wt = shifted_QKV.shape[2:]
		# partition windows
		shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
		qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C
		attn_windows = self.attn(qkv)
		shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C
		out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
		attn_out = out.permute(0, 3, 1, 2)
		out = self.proj(attn_out)
		return out

#transformer块设计
class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 window_size=8, shift_size=0):
        super().__init__()
        # 前置归一化层
        self.norm1 = norm_layer(dim)
        # 注意力模块
        self.attn = SEMHSA(network_depth,dim,
                          num_heads=num_heads,
                          window_size=window_size,
                          shift_size=shift_size)
        # 后置归一化层
        self.norm2 = norm_layer(dim)
        # 多层感知机模块
        self.mlp = SparseMLP(network_depth,dim,
                            hidden_channels=int(dim * mlp_ratio),
                            out_channels=dim)
    def forward(self, x):
        # 残差连接
        x = x + self.attn(self.norm1(x))  # 注意力分支
        x = x + self.mlp(self.norm2(x))

        return x

class BasicLayer(nn.Module):
	def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
				 norm_layer=nn.BatchNorm2d, window_size=8, ):
		super().__init__()
		self.dim = dim
		self.depth = depth #构建循环的transformer的循环深度
		self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2, #窗口变换  #shift_size=0（标准窗口划分）·偶数 #shift_size=window_size//2（滑动窗口划分）·奇数
                            )
            for i in range(depth)])
	def forward(self, x):
		for blk in self.blocks:
			x = blk(x)
		return x


# 特征融合模块
class FeatureFusion(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.kernel_sizes = [3, 5, 7] #多尺度特征
		self.num_scales = len(self.kernel_sizes)
		# 编码器单独处理
		self.encoder_branches = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2),
				nn.BatchNorm2d(out_channels),
				nn.GELU(),
			) for k in self.kernel_sizes
		])
		# 解码器单独处理
		self.decoder_branches = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2),
				nn.BatchNorm2d(out_channels),
				nn.GELU	(),
			) for k in self.kernel_sizes
		])

		self.PDSCA = PDSCA(in_channels=out_channels, reduction=out_channels // 2)
		# self.PDSCA2 = PDSCA(in_channels=out_channels, reduction=out_channels // 2)

		self.cross_attention = nn.ModuleDict({
			'enc2dec': self.PDSCA,  # 编码器->解码器方向
			'dec2enc': self.PDSCA   # 解码器->编码器方向
		})


		# self.conv_fusion = nn.Sequential(
		# 	nn.Conv2d(in_channels * 2 , out_channels , 3, padding=1),
		# 	nn.BatchNorm2d(out_channels),
		# 	nn.GELU(),
		# 	nn.Conv2d(out_channels, out_channels, 1),
		# 	nn.BatchNorm2d(out_channels),
		# 	nn.GELU()
		# )

	def forward(self, enc_feat, dec_feat):
		# 多尺度特征提取 -------------------------------
		enc_scales, dec_scales = [], []
		for enc_branch, dec_branch in zip(self.encoder_branches, self.decoder_branches):
			# 编码器分支处理
			enc_main = enc_branch(enc_feat)
			enc_scales.append(enc_main)
			# 解码器分支处理
			dec_main = dec_branch(dec_feat)
			dec_scales.append(dec_main)

		enc_scales_all = sum(enc_scales)
		dec_scales_all = sum(dec_scales)
		# 跨尺度交互融合 ------------------------------
		# 编码器->解码器注意力
		attn_dec = self.cross_attention['enc2dec'](enc_scales_all)
		# 解码器->编码器注意力
		attn_enc = self.cross_attention['dec2enc'](dec_scales_all)

		# 多尺度特征融合
		multi_scale = attn_enc*enc_scales_all+attn_dec*dec_scales_all
		# 最终融合与残差连接
		return multi_scale+enc_feat+ dec_feat

class SEMHSADahazeFormer(nn.Module):
	def __init__(self,
				 mode,
				 in_chans=3,  # 输入图像RGB3通道
				 out_chans=3,  # 输入通道
				 window_size=8,
				 embed_dims=[32, 64, 128, 256, 128, 64, 32],  # [嵌入, 下采样1, 下采样2, 上采样1, 上采样2，维度恢复]特征维度
				 mlp_ratios=[2., 2., 4., 6., 4., 2., 2.],  # MLP扩展比率（各阶段MLP隐藏层维度倍数）
				 depths=[2, 2, 4, 6, 4, 2, 2],  # 重叠taransformer块数量
				 num_heads=[1, 2, 4, 8, 4, 2, 1],  # 各阶段的注意力头数量
				 norm_layer=nn.BatchNorm2d):

		super(SEMHSADahazeFormer, self).__init__()
		self.window_size = window_size  # 注意力计算窗口
		self.mlp_ratios = mlp_ratios  # MLP扩展系数
		#颜色空间转换，目的是分离亮度空间和色度空间
		self.rgb2lab =  RGBLABConverter()

		self.linght_conv = nn.Conv2d(1, embed_dims[0], 3, padding=1)
		self.ab_conv = nn.Conv2d(2, embed_dims[0], 3, padding=1)
		self.patch_embed = PatchEmbed(patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)
		self.skip0_0 =nn.Sequential(
			nn.Conv2d(2, 24, 3, padding=1,bias=False),
            nn.BatchNorm2d(24),
			nn.GELU(),
			nn.Conv2d(24,in_chans, 1),
		)
		#self.skip1_2 = nn.Conv2d(embed_dims[0], embed_dims[0], 1,bias=False)


		# 编码器阶段 --------------------------------------------------
		''''阶段一：原始分辨率-->1/2分辨率处理'''
		self.encoder_level1 = BasicLayer(
			network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
            num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
            norm_layer=norm_layer,window_size=window_size
           )
		self.down1_2 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
		self.fusion1 = FeatureFusion(embed_dims[0] , embed_dims[6])
		#self.skip2_3 = nn.Conv2d(embed_dims[1], embed_dims[1], 1,bias=False)
		'''阶段二：1/2分辨率处理-->1/4分辨率处理'''
		self.encoder_level2 = BasicLayer(
			network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
			num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
			norm_layer=norm_layer,window_size=window_size
		)
		self.down2_3 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2]) #48->96
		#self.skip3_4 =nn.Conv2d(embed_dims[2], embed_dims[2], 1)# 跳跃连接保持维度【96】
		self.fusion2 = FeatureFusion(embed_dims[1] , embed_dims[5])
		''''阶段三：1/4分辨率处理-->1/8分辨率处理'''
		self.encoder_level3 = BasicLayer(
			network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
			num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],norm_layer=norm_layer,
			window_size=window_size)
		# 下采样与跳跃连结
		self.down3_4 = PatchEmbed(patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3]) #9
		self.fusion3 = FeatureFusion(embed_dims[4] , embed_dims[4])
		# 中间平静层阶段 --------------------------------------------------
		self.bottleneck = BasicLayer(
			network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
			num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], norm_layer=norm_layer,
			window_size=window_size)
       # 解编码器阶段 --------------------------------------------------
		''''阶段一：1/4分辨率重建【192->96】'''
		self.up4_3 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
		self.decoder_level1 = BasicLayer(
			network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
			num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
			norm_layer=norm_layer, window_size=window_size,
			)
		''''阶段二：1/2分辨率重建【96->48】'''
		self.up3_2 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[5], embed_dim=embed_dims[4])
		self.decoder_level2 = BasicLayer(
			network_depth=sum(depths), dim=embed_dims[5], depth=depths[5],
			num_heads=num_heads[5], mlp_ratio=mlp_ratios[5],
			norm_layer=norm_layer, window_size=window_size,
		)
		''''阶段三：原始分辨率重建【48->24->3】'''
		self.up2_1 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[6], embed_dim=embed_dims[5])
		self.decoder_level3 = BasicLayer(
			network_depth=sum(depths), dim=embed_dims[6], depth=depths[6],
			num_heads=num_heads[6], mlp_ratio=mlp_ratios[6],
			norm_layer=norm_layer, window_size=window_size,
		)
		self.up1_0 = PatchUnEmbed(
			patch_size=1, out_chans=out_chans, embed_dim=embed_dims[6], kernel_size=3)
		#self.sigmoid = nn.Sigmoid()
		self.final = nn.Sequential(
			nn.ReflectionPad2d(3),  # 镜像填充保持尺寸
			nn.Conv2d(out_chans,out_chans, 7, padding=0, bias=True),
			nn.Sigmoid()
		)
		#尺寸对其与边界处理
	def check_image_size(self, x):
		# NOTE: for I2I test
		_, _, h, w = x.size()
		mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
		mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
		x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
		return x


	def forward(self, x):
		# 输入尺寸检查与补边
		x = self.check_image_size(x)
		lab = self.rgb2lab(x,mode='rgb2lab')
		l_channel = lab[:, :1, :, :]  # 亮度通道
		ab_channels = lab[:, 1:, :, :]  # 色度通道
		#检查补边
		#skip = x

        #注意是否需要加入原始的rgb信息
		x = self.linght_conv (l_channel) + self.ab_conv(ab_channels)+ self.patch_embed(x)
		#初始嵌入
		skip0 = x

		#编码器阶段 --------------------------------------------------
		x = self.encoder_level1(x)
		x = self.down1_2(x)
		skip1 = x
		x = self.encoder_level2(x)
		x = self.down2_3(x)
		skip2 = x

		x = self.encoder_level3(x)
		x = self.down3_4(x) #96->192
		# 中间平静层阶段 --------------------------------------------------
		x = self.bottleneck(x)

		# 解码器阶段 --------------------------------------------------
		x = self.up4_3(x)  # 192->96
		#skip2_aligned = self.skip3_4(skip2)
		x = self.fusion3(skip2,x)
		x = self.decoder_level1(x)


		x = self.up3_2(x)
		#skip1_aligned = self.skip2_3(skip1)
		x = self.fusion2(skip1,x)
		x = self.decoder_level2(x)


		x = self.up2_1(x)
		#skip0_aligned = self.skip1_2(skip0)
		x = self.fusion1(skip0,x)
		x = self.decoder_level3(x)

		x = self.up1_0(x)
		# 原始跳跃
		skip_aligned = self.skip0_0(ab_channels)
		return self.final(x+skip_aligned)

#计算参数量
def count_param(model):
	param_count = 0
	for param in model.parameters():
		param_count += param.view(-1).size()[0]
	return param_count

def build_net(mode):
    return SEMHSADahazeFormer(mode)



if __name__ == "__main__":
	import time
	from thop import profile
	# ====================== 模型初始化 ======================
	model = SEMHSADahazeFormer(
		mode = 1,
		in_chans=3,
		out_chans=3,
		window_size=16,
		embed_dims=[32, 64, 128, 256, 128, 64, 32],
		depths=[2, 2, 4, 6, 4, 2, 2]
	)

	# ====================== 参数量统计 ======================
	print("\n" + "=" * 50 + " 参数量 " + "=" * 50)
	total_params = count_param(model)
	print(f"总参数: {total_params / 1e6:.2f}M")

	print("\n" + "=" * 50 + " FLOPs " + "=" * 50)
	input = torch.randn(1, 3,256, 256)  # 示例输入（调整尺寸为8的倍数）
	model.eval()  # 确保模型在评估模式

	flops, params = profile(model, inputs=(input,))
	print(f"FLOPs: {flops / 1e9:.2f}G")

	# ====================== 验证一致性 ======================
	print("\n" + "=" * 50 + " 一致性检查 " + "=" * 50)
	print(f"Params结果对比：")
	print(f"- count_param函数: {total_params / 1e6:.2f}M")
	print(f"- thop库结果: {params / 1e6:.2f}M")

