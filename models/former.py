import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import math
from  models.PrecisionDrivenSynergisticContextualAttention import PDSCA
from  models.ColorSpaceConverter import RGBLABConverter

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
		B, C, H, W = x.shape
		x = self.activation(self.BN(x))
		x_h = x.permute(0, 3, 1, 2)  # [B, W, C, H]
		x_h = x_h.reshape(B * W, C, H)  # [B*W, C, H]
		x_h = self.proj_h(x_h)  # [B*W, C, H]
		x_h = x_h.reshape(B, W, C, H)  # [B, W, C, H]
		x_h = x_h.permute(0, 2, 3, 1)  # [B, C, H, W]
		x_w = x.permute(0, 2, 1, 3)  # [B, H, C, W]
		x_w = x_w.reshape(B * H, C, W)  # [B*H, C, W]
		x_w = self.proj_w(x_w)  # [B*H, C, W]
		x_w = x_w.reshape(B, H, C, W)  # [B, H, C, W]
		x_w = x_w.permute(0, 2, 1, 3)  # [B, C, H, W]
		fused = torch.cat([x, x_h, x_w], dim=1)  # [B, 3C, H, W]
		return self.fuse(fused)  # [B, C, H, W]


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.PixelUnshuffle(patch_size),
            nn.Conv2d(in_chans * (patch_size**2), embed_dim, kernel_size=1)
        )

    def forward(self, x):
        return self.proj(x)


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

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

def window_partition(x, window_size):
	B, H, W, C = x.shape
	x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
	windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size**2, C)

	return windows


def window_reverse(windows, window_size, H, W):
	B = int(windows.shape[0] / (H * W / window_size / window_size))
	x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
	x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
	return x


class get_TD_positions(nn.Module):
	def __init__(self, dim=2, window_size=7):
		super().__init__()
		self.window_size = window_size

		coords_h = torch.linspace(-1., 1., steps=window_size)
		coords_w = torch.linspace(-1., 1., steps=window_size)
		y_grid, x_grid = torch.meshgrid(coords_h, coords_w, indexing='ij')
		grid = torch.stack([x_grid, y_grid], dim=0)  # [2, Wh, Ww]
		self.register_buffer("grid", grid.unsqueeze(0))  # [1, 2, Wh, Ww]
		self.conv = nn.Conv2d(2, dim, kernel_size=1)
		self.scale = nn.Parameter(torch.tensor(0.1))

	def forward(self):

		pe = self.conv(self.grid)
		pe_flatten = pe.view(1, pe.size(1), -1)
		relative_pe = pe_flatten.permute(0, 2, 1)[:, :, None, :] - \
					  pe_flatten.permute(0, 2, 1)[:, None, :, :]
		relative_pe = relative_pe.squeeze(0)  # [Wh*Ww, Wh*Ww, dim]
		return self.scale * relative_pe


class WindowAttention(nn.Module):
	def __init__(self, dim, window_size, num_heads, reduction=4,  bias=False):

		super().__init__()
		self.dim = dim
		self.window_size = window_size  # Wh, Ww
		self.num_heads = num_heads
		self.head_dim = dim // num_heads
		assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"
		self.log_temp = nn.Parameter(torch.tensor([math.log(0.1)]))

		self.register_buffer('num_heads_tensor', torch.tensor(num_heads, dtype=torch.float32))
		self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
		self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3,
									bias=bias)  #

		self.pdcsa_q = PDSCA(in_channels=self.head_dim, reduction=reduction)
		self.pdcsa_k = PDSCA(in_channels=self.head_dim, reduction=reduction)
		self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
		self.positions = get_TD_positions(dim=self.dim, window_size=self.window_size)

		self.meta = nn.Sequential(
			nn.Linear(self.dim, 128),
			nn.GELU(),
			nn.Linear(128,self.num_heads)
		)												  #[Wh * Ww, Wh * Ww, num_heads]
		self.softmax = nn.Softmax(dim=-1)


	def forward(self, qkv):

		B_, N, _ = qkv.shape
		qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

		q, k, v = qkv[0], qkv[1], qkv[2]
		temperature = torch.exp(self.log_temp)
		mask_q = self.pdcsa_q(q.reshape(-1, self.head_dim, self.window_size, self.window_size))
		mask_k = self.pdcsa_k(k.reshape(-1, self.head_dim, self.window_size, self.window_size))
		q = q * mask_q.view_as(q).sigmoid()
		k = k * mask_k.view_as(k).sigmoid()
		attn = (q @ k.transpose(-2, -1)) * temperature
		positions = self.positions().to(self.log_temp.device)
		position_bias = self.meta(positions).permute(2, 0, 1).contiguous()
		attn = attn + position_bias.unsqueeze(0)
		attn = self.softmax(attn)
		out = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim) # [B, H, N, D]
		out = self.project_out(out.permute(0, 2, 1).unsqueeze(-1)).squeeze(-1).permute(0, 2, 1)

		return out

class SEMHSA(nn.Module):
	def __init__(self, network_depth, dim, num_heads, window_size, shift_size):
		super().__init__()
		self.dim = dim
		self.head_dim = int(dim // num_heads)
		self.num_heads = num_heads

		self.window_size = window_size
		self.shift_size = shift_size
		self.network_depth = network_depth


		self.QK = nn.Conv2d(dim, dim * 2, 1)
		self.V = nn.Conv2d(dim, dim, 1)
		self.attn = WindowAttention(dim, window_size, num_heads)
		self.proj = nn.Conv2d(dim, dim, 1)
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
		shifted_QKV = self.check_size(QKV, self.shift_size > 0)
		Ht, Wt = shifted_QKV.shape[2:]
		shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
		qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C
		attn_windows = self.attn(qkv)
		shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C
		out = shifted_out[:, self.shift_size:(self.shift_size + H), self.shift_size:(self.shift_size + W), :]
		attn_out = out.permute(0, 3, 1, 2)
		out = self.proj(attn_out)
		return out

class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 window_size=8, shift_size=0):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = SEMHSA(network_depth,dim,
                          num_heads=num_heads,
                          window_size=window_size,
                          shift_size=shift_size)
        self.norm2 = norm_layer(dim)
        self.mlp = SparseMLP(network_depth,dim,
                            hidden_channels=int(dim * mlp_ratio),
                            out_channels=dim)
    def forward(self, x):
        # 残差连接
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class BasicLayer(nn.Module):
	def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
				 norm_layer=nn.BatchNorm2d, window_size=8, ):
		super().__init__()
		self.dim = dim
		self.depth = depth
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

class FeatureFusion(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.kernel_sizes = [3, 5, 7]
		self.num_scales = len(self.kernel_sizes)
		self.encoder_branches = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2),
				nn.BatchNorm2d(out_channels),
				nn.GELU(),
			) for k in self.kernel_sizes
		])
		self.decoder_branches = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k // 2),
				nn.BatchNorm2d(out_channels),
				nn.GELU	(),
			) for k in self.kernel_sizes
		])

		self.PDSCA = PDSCA(in_channels=out_channels, reduction=out_channels // 2)

		self.cross_attention = nn.ModuleDict({
			'enc2dec': self.PDSCA,  # 编码器->解码器方向
			'dec2enc': self.PDSCA   # 解码器->编码器方向
		})
	def forward(self, enc_feat, dec_feat):
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
		attn_dec = self.cross_attention['enc2dec'](enc_scales_all)
		attn_enc = self.cross_attention['dec2enc'](dec_scales_all)
		multi_scale = attn_enc*enc_scales_all+attn_dec*dec_scales_all
		return multi_scale+enc_feat+ dec_feat

class SEMHSADahazeFormer(nn.Module):
	def __init__(self,
				 mode,
				 in_chans=3,  # 输入图像RGB3通道
				 out_chans=3,  # 输入通道
				 window_size=8,
				 embed_dims=[32, 64, 128, 256, 128, 64, 32],
				 mlp_ratios=[2., 2., 4., 6., 4., 2., 2.],
				 depths=[2, 2, 4, 6, 4, 2, 2],
				 num_heads=[1, 2, 4, 8, 4, 2, 1],
				 norm_layer=nn.BatchNorm2d):

		super(SEMHSADahazeFormer, self).__init__()
		self.window_size = window_size
		self.mlp_ratios = mlp_ratios
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
		self.encoder_level1 = BasicLayer(
			network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
            num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
            norm_layer=norm_layer,window_size=window_size
           )
		self.down1_2 = PatchEmbed(patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
		self.fusion1 = FeatureFusion(embed_dims[0] , embed_dims[6])
		self.encoder_level2 = BasicLayer(
			network_depth=sum(depths), dim=embed_dims[1], depth=depths[1],
			num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
			norm_layer=norm_layer,window_size=window_size
		)
		self.down2_3 = PatchEmbed(patch_size=2, in_chans=embed_dims[1], embed_dim=embed_dims[2]) #
		self.fusion2 = FeatureFusion(embed_dims[1] , embed_dims[5])
		self.encoder_level3 = BasicLayer(
			network_depth=sum(depths), dim=embed_dims[2], depth=depths[2],
			num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],norm_layer=norm_layer,
			window_size=window_size)
		self.down3_4 = PatchEmbed(patch_size=2, in_chans=embed_dims[2], embed_dim=embed_dims[3]) #9
		self.fusion3 = FeatureFusion(embed_dims[4] , embed_dims[4])
		self.bottleneck = BasicLayer(
			network_depth=sum(depths), dim=embed_dims[3], depth=depths[3],
			num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], norm_layer=norm_layer,
			window_size=window_size)
		self.up4_3 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[4], embed_dim=embed_dims[3])
		self.decoder_level1 = BasicLayer(
			network_depth=sum(depths), dim=embed_dims[4], depth=depths[4],
			num_heads=num_heads[4], mlp_ratio=mlp_ratios[4],
			norm_layer=norm_layer, window_size=window_size,
			)
		self.up3_2 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[5], embed_dim=embed_dims[4])
		self.decoder_level2 = BasicLayer(
			network_depth=sum(depths), dim=embed_dims[5], depth=depths[5],
			num_heads=num_heads[5], mlp_ratio=mlp_ratios[5],
			norm_layer=norm_layer, window_size=window_size,
		)
		self.up2_1 = PatchUnEmbed(patch_size=2, out_chans=embed_dims[6], embed_dim=embed_dims[5])
		self.decoder_level3 = BasicLayer(
			network_depth=sum(depths), dim=embed_dims[6], depth=depths[6],
			num_heads=num_heads[6], mlp_ratio=mlp_ratios[6],
			norm_layer=norm_layer, window_size=window_size,
		)
		self.up1_0 = PatchUnEmbed(
			patch_size=1, out_chans=out_chans, embed_dim=embed_dims[6], kernel_size=3)
		self.final = nn.Sequential(
			nn.ReflectionPad2d(3),  # 镜像填充保持尺寸
			nn.Conv2d(out_chans,out_chans, 7, padding=0, bias=True),
			nn.Sigmoid()
		)
	def check_image_size(self, x):
		_, _, h, w = x.size()
		mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
		mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
		x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
		return x


	def forward(self, x):
		x = self.check_image_size(x)
		lab = self.rgb2lab(x,mode='rgb2lab')
		l_channel = lab[:, :1, :, :]  # 亮度通道
		ab_channels = lab[:, 1:, :, :]  # 色度通道
		x = self.linght_conv (l_channel) + self.ab_conv(ab_channels)+ self.patch_embed(x)
		skip0 = x
		x = self.encoder_level1(x)
		x = self.down1_2(x)
		skip1 = x
		x = self.encoder_level2(x)
		x = self.down2_3(x)
		skip2 = x

		x = self.encoder_level3(x)
		x = self.down3_4(x)
		x = self.bottleneck(x)
		x = self.up4_3(x)
		x = self.fusion3(skip2,x)
		x = self.decoder_level1(x)
		x = self.up3_2(x)
		x = self.fusion2(skip1,x)
		x = self.decoder_level2(x)
		x = self.up2_1(x)
		x = self.fusion1(skip0,x)
		x = self.decoder_level3(x)
		x = self.up1_0(x)
		skip_aligned = self.skip0_0(ab_channels)
		return self.final(x+skip_aligned)


def count_param(model):
	param_count = 0
	for param in model.parameters():
		param_count += param.view(-1).size()[0]
	return param_count

def build_net(mode):
    return SEMHSADahazeFormer(mode)

