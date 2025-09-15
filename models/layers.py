import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Borrowed from ''Improving image restoration by revisiting global information aggregation''
# --------------------------------------------------------------------------------
train_size = (1,3,256,256)#初始化训练尺寸，方便之后做分辨率调整
class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        """
        自适应平均池化层初始化
        Args:
            kernel_size: 池化核尺寸 (默认None)
            base_size: 基准尺寸，用于动态计算kernel_size
            auto_pad: 自动填充保持特征图尺寸 (默认True)
            fast_imp: 启用快速实现模式 (默认False)
        """
        super().__init__()
        self.kernel_size = kernel_size    # 实际使用的池化核尺寸
        self.base_size = base_size        # 训练时基准尺寸(用于动态调整)
        self.auto_pad = auto_pad          # 自动填充标志
        
        # 快速实现相关参数
        self.fast_imp = fast_imp          # 是否启用快速近似实现
        self.rs = [5,4,3,2,1]             # 快速实现的降采样率候选值
        self.max_r1 = self.rs[0]          # 高度方向最大降采样率
        self.max_r2 = self.rs[0]          # 宽度方向最大降采样率

    def extra_repr(self) -> str:
        """ 
        生成模块的附加表示信息
        用于打印模块时显示关键配置参数
        返回格式: kernel_size, base_size, stride, fast_imp
        """
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size,  # 当前使用的池化核尺寸
            self.base_size,    # 基准尺寸(用于动态调整)
            self.kernel_size,  # 步幅与核尺寸相同
            self.fast_imp      # 快速实现模式标志
        )

    def forward(self, x):
        """
        前向传播实现自适应平均池化
        包含两种实现方式：快速近似实现和标准实现
        """
        # 动态计算kernel_size (当使用base_size时)
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            # 根据输入尺寸动态调整kernel_size
            self.kernel_size[0] = x.shape[2]*self.base_size[0]//train_size[-2]  # 高度方向
            self.kernel_size[1] = x.shape[3]*self.base_size[1]//train_size[-1]  # 宽度方向
            
            # 计算快速实现的最大降采样率
            self.max_r1 = max(1, self.rs[0]*x.shape[2]//train_size[-2])
            self.max_r2 = max(1, self.rs[0]*x.shape[3]//train_size[-1])

        # 快速近似实现路径
        if self.fast_imp:   # 非等效实现但速度更快
            h, w = x.shape[2:]
            if self.kernel_size[0]>=h and self.kernel_size[1]>=w:  # 全剧池化
                out = F.adaptive_avg_pool2d(x,1)
            else:
                # 寻找可整除的降采样率
                r1 = [r for r in self.rs if h%r==0][0]  # 高度方向降采样率
                r2 = [r for r in self.rs if w%r==0][0]  # 宽度方向降采样率
                r1 = min(self.max_r1, r1)  # 限制最大降采样率
                r2 = min(self.max_r2, r2)
                
                # 降采样+积分图计算局部均值
                s = x[:,:,::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h-1, self.kernel_size[0]//r1), min(w-1, self.kernel_size[1]//r2)
                out = (s[:,:,:-k1,:-k2]-s[:,:,:-k1,k2:]-s[:,:,k1:,:-k2]+s[:,:,k1:,k2:])/(k1*k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1,r2))  # 上采样恢复尺寸
        else:  # 标准实现路径
            n, c, h, w = x.shape
            # 积分图计算
            s = x.cumsum(dim=-1).cumsum(dim=-2)
            s = torch.nn.functional.pad(s, (1,0,1,0))  # 填充便于计算
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])
            # 滑动窗口求和计算
            s1, s2, s3, s4 = s[:,:,:-k1,:-k2],s[:,:,:-k1,k2:], s[:,:,k1:,:-k2], s[:,:,k1:,k2:]
            out = s4+s1-s2-s3
            out = out / (k1*k2)  # 求平均
    
        # 自动填充保持输入输出尺寸一致
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w)//2, (w - _w + 1)//2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')  # 镜像填充
        
        return out
# --------------------------------------------------------------------------------



class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        """
        基础卷积模块初始化
        Args:
            in_channel: 输入通道数
            out_channel: 输出通道数
            kernel_size: 卷积核尺寸
            stride: 卷积步长
            bias: 是否启用偏置 (默认True)
            norm: 是否添加批归一化 (默认False)
            relu: 是否添加激活函数 (默认True)
            transpose: 是否使用转置卷积 (默认False)
        """
        super(BasicConv, self).__init__()
        # 当同时使用norm和bias时，禁用bias（BN包含偏置项）
        if bias and norm:
            bias = False

        padding = kernel_size // 2  # 普通卷积的标准填充计算
        layers = list()
        if transpose:  # 转置卷积模式
            padding = kernel_size // 2 -1  # 转置卷积的特殊填充计算
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:  # 普通卷积模式
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))  # 批归一化层
        if relu:
            layers.append(nn.GELU())  # GELU激活函数
        self.main = nn.Sequential(*layers)  # 构建序列模块

    def forward(self, x):
        return self.main(x)



class Gap(nn.Module):
    def __init__(self, in_channel, mode) -> None:
        """
        全局自适应池化模块初始化
        Args:
            in_channel: 输入通道数
            mode: 模式配置元组 (训练/测试, 场景类型)
        """
        super().__init__()
        # 初始化可学习的特征缩放参数
        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)  # 低频分量缩放因子
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)  # 高频分量缩放因子

        # 根据模式选择池化策略
        if mode[0] == 'train':  # 训练模式
            self.gap = nn.AdaptiveAvgPool2d((1,1))  # 标准自适应池化
        elif mode[0] == 'test':  # 测试模式
            # 根据场景类型选择不同配置
            if mode[1] == 'Indoor':    # 室内场景
                self.gap = AvgPool2d(base_size=246)  # 基于246基准尺寸的动态池化
            elif mode[1] == 'Outdoor': # 室外场景
                self.gap = AvgPool2d(base_size=210)  # 基于210基准尺寸的动态池化

    def forward(self, x):
        """
        特征分解与融合前向传播
        将输入分解为低频和高频分量，并进行自适应缩放
        Args:
            x: 输入特征图，形状为[B, C, H, W]
        Returns:
            融合后的特征图
        """
        x_d = self.gap(x)  # 全局平均池化获取低频分量
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)  # 计算高频残差并缩放
        x_d = x_d  * self.fscale_d[None, :, None, None]  # 低频分量缩放
        return x_d + x_h  # 特征融合

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, mode, filter=False):
        """
        残差块模块初始化
        Args:
            in_channel: 输入通道数
            out_channel: 输出通道数
            mode: 模式配置元组 (训练/测试, 场景类型)
            filter: 是否启用动态滤波 (默认False)
        """
        super(ResBlock, self).__init__()
        # 构建双卷积基础结构
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True)  # 带激活的卷积
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)  # 无激活的卷积
        
        # 动态滤波相关配置
        self.filter = filter  # 是否启用动态滤波标志
        if filter:  # 当启用动态滤波时
            # 创建3x3和5x5两种动态滤波器
            self.dyna = dynamic_filter(in_channel//2, mode)  # 3x3卷积核的动态滤波
            self.dyna_2 = dynamic_filter(in_channel//2, mode, kernel_size=5)  # 5x5卷积核的动态滤波
        else:  # 禁用动态滤波时使用恒等映射
            self.dyna = nn.Identity()
            self.dyna_2 = nn.Identity()

        # 注意力模块配置
        self.localap = Patch_ap(mode, in_channel//2, patch_size=2)  # 局部区域注意力
        self.global_ap = Gap(in_channel//2, mode)  # 全局注意力


    def forward(self, x):
        """
        残差块前向传播过程
        实现包含动态滤波和双注意力机制的特征处理
        Args:
            x: 输入特征图，形状为[B, C, H, W]
        Returns:
            残差连接后的输出特征图
        """
        # 第一阶段卷积处理
        out = self.conv1(x)
        
        # 动态滤波处理分支
        if self.filter:
            k3, k5 = torch.chunk(out, 2, dim=1)  # 沿通道维度分为两部分
            out_k3 = self.dyna(k3)  # 3x3动态滤波处理
            out_k5 = self.dyna_2(k5)  # 5x5动态滤波处理
            out = torch.cat((out_k3, out_k5), dim=1)  # 合并滤波结果
            
        # 双注意力机制处理
        non_local, local = torch.chunk(out, 2, dim=1)  # 再次分割特征图
        non_local = self.global_ap(non_local)  # 全局注意力处理
        local = self.localap(local)  # 局部注意力处理
        out = torch.cat((non_local, local), dim=1)  # 合并注意力特征
        
        # 第二阶段卷积处理
        out = self.conv2(out)
        # 残差连接
        return out + x



class dynamic_filter(nn.Module):
    def __init__(self, inchannels, mode, kernel_size=3, stride=1, group=8):
        """
        动态滤波器模块初始化
        Args:
            inchannels: 输入特征通道数
            mode: 模式配置元组 (训练/测试, 场景类型)
            kernel_size: 动态卷积核尺寸 (默认3)
            stride: 卷积步长 (默认1)
            group: 分组卷积数 (默认8)
        """
        super(dynamic_filter, self).__init__()
        # 基础参数配置
        self.stride = stride  # 卷积步长
        self.kernel_size = kernel_size  # 动态滤波器尺寸
        self.group = group  # 分组卷积数
        
        # 可学习参数初始化
        self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)  # 低频分量缩放因子
        self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)  # 高频分量缩放因子

        # 动态卷积核生成器
        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)  # 生成卷积核参数
        self.bn = nn.BatchNorm2d(group*kernel_size**2)  # 卷积核参数归一化
        self.act = nn.Softmax(dim=-2)  # 卷积核权重归一化
        
        # 参数初始化
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')  # He初始化

        # 特征处理组件
        self.pad = nn.ReflectionPad2d(kernel_size//2)  # 反射填充保持尺寸
        self.ap = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.modulate = SFconv(inchannels, mode)  # 特征调制模块

    def forward(self, x):
        identity_input = x  # 保存原始输入用于残差连接
        # 生成动态卷积核参数
        low_filter = self.ap(x)  # 全局平均池化获取空间信息
        low_filter = self.conv(low_filter)  # 生成动态卷积核参数
        low_filter = self.bn(low_filter)  # 参数归一化处理

        # 特征展开与重塑
        n, c, h, w = x.shape  
        # 展开输入特征进行卷积操作 [B, C, H, W] -> [B, G, C/G, K^2, H*W]
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        # 动态卷积核处理
        n,c1,p,q = low_filter.shape
        # 重塑卷积核参数 [B, G*K^2, 1, 1] -> [B, G, K^2, H*W]
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
       
        low_filter = self.act(low_filter)  # Softmax归一化卷积核权重
    
        # 动态卷积计算（逐位置卷积）
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)  # 加权求和得到低频分量

        # 高频分量计算
        out_high = identity_input - low_part  # 原始输入减去低频得到高频残差
        
        # 特征调制与融合
        out = self.modulate(low_part, out_high)  # SFconv模块调制高低频特征
        return out



class SFconv(nn.Module):
    def __init__(self, features, mode, M=2, r=2, L=32) -> None:
        """
        特征调制模块初始化（SFconv）
        Args:
            features: 输入特征通道数
            mode: 模式配置元组 (训练/测试, 场景类型)
            M: 注意力分支数 (默认2)
            r: 通道压缩比 (默认2)
            L: 通道数下限 (默认32)
        """
        super().__init__()
        # 中间通道数计算（取压缩后通道与下限的最大值）
        d = max(int(features/r), L)
        self.features = features

        # 特征压缩层
        self.fc = nn.Conv2d(features, d, 1, 1, 0)  # 1x1卷积降维
        
        # 多分支注意力生成器
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)  # 1x1卷积生成注意力图
            )
        
        # 注意力处理组件
        self.softmax = nn.Softmax(dim=1)  # 通道维度归一化
        self.out = nn.Conv2d(features, features, 1, 1, 0)  # 最终融合层

        # 池化策略配置
        if mode[0] == 'train':
            self.gap = nn.AdaptiveAvgPool2d(1)  # 训练时标准池化
        elif mode[0] == 'test':
            if mode[1] == 'Indoor':
                self.gap = AvgPool2d(base_size=246)  # 室内场景动态池化
            elif mode[1] == 'Outdoor':
                self.gap = AvgPool2d(base_size=210)  # 室外场景动态池化

    def forward(self, low, high):
        """
        特征调制前向传播
        实现高低频特征的注意力加权融合
        Args:
            low: 低频特征图，形状为[B, C, H, W]
            high: 高频特征图，形状同低频
        Returns:
            调制后的融合特征
        """
        emerge = low + high  # 初步融合特征
        emerge = self.gap(emerge)  # 全局空间信息压缩

        fea_z = self.fc(emerge)  # 特征压缩到低维空间

        # 生成双路注意力权重
        high_att = self.fcs[0](fea_z)  # 高频注意力分支
        low_att = self.fcs[1](fea_z)   # 低频注意力分支
        
        # 拼接并归一化注意力向量
        attention_vectors = torch.cat([high_att, low_att], dim=1)  # 通道维度拼接
        attention_vectors = self.softmax(attention_vectors)  # 通道维度归一化

        # 分割注意力向量
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)  # 分割回双路注意力

        # 特征调制过程
        fea_high = high * high_att  # 高频特征加权
        fea_low = low * low_att     # 低频特征加权
        
        # 特征融合与输出
        out = self.out(fea_high + fea_low)  # 加权特征融合
        return out



class Patch_ap(nn.Module):
    def __init__(self, mode, inchannel, patch_size):
        """
        局部区域注意力模块初始化
        Args:
            mode: 模式配置元组 (训练/测试, 场景类型)
            inchannel: 输入特征通道数
            patch_size: 分块处理尺寸
        """
        super(Patch_ap, self).__init__()

        # 分块处理参数
        self.patch_size = patch_size  # 特征分块尺寸
        self.channel = inchannel * patch_size**2  # 分块后通道数(原始通道×块内像素数)
        
        # 可学习缩放参数
        self.h = nn.Parameter(torch.zeros(self.channel))  # 高频分量缩放因子
        self.l = nn.Parameter(torch.zeros(self.channel))  # 低频分量缩放因子

    def forward(self, x):
        """
        局部区域注意力前向传播
        实现特征图分块处理与高低频特征融合
        Args:
            x: 输入特征图，形状为[B, C, H, W]
        Returns:
            处理后的特征图，保持原始空间维度
        """
        # 特征图分块处理
        # [B, C, H, W] -> [B, C, P1, W1, P2, W2] (P=块大小，W=块内位置)
        patch_x = rearrange(x, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size)
        # [B, C, P1, W1, P2, W2] -> [B, C*P1*P2, W1, W2] (合并块内像素到通道维度)
        patch_x = rearrange(patch_x, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size)

        # 高低频特征分解
        low = self.ap(patch_x)  # 低频分量(空间池化)
        high = (patch_x - low) * self.h[None, :, None, None]  # 高频残差缩放
        
        # 特征融合与重建
        out = high + low * self.l[None, :, None, None]  # 低频分量缩放并融合
        # 恢复原始空间排列 [B, C*P1*P2, W1, W2] -> [B, C, H, W]
        out = rearrange(out, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size, p2=self.patch_size)

        return out
