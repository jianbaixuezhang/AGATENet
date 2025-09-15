import torch
import torch.nn as nn
from models.former import SEMHSADahazeFormer
import os

#专家实际就是已经训练的网络
class Expert(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.model = SEMHSADahazeFormer(*args, **kwargs)
    def forward(self, x):
        return self.model(x)

#门控网络用于产生权重
class GatingNetwork(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        # 增强的特征提取
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts)
        )
        
    def forward(self, x):
        return torch.softmax(self.feature_extractor(x), dim=1)

class MoEModel(nn.Module):
    def __init__(self, expert_args_list, pretrained_paths=None):
        super().__init__()
        self.num_experts = len(expert_args_list)
        self.experts = nn.ModuleList([
            Expert(**expert_args) for expert_args in expert_args_list
        ])
        self.gate = GatingNetwork(self.num_experts)
        
        if pretrained_paths is not None:
            self.load_pretrained_experts(pretrained_paths)
        # 冻结专家参数
        for p in self.experts.parameters():
            p.requires_grad = False

    def load_pretrained_experts(self, paths):
        """
        加载预训练的专家模型权重（.pkl文件）
        Args:
            paths: 专家模型权重文件路径列表（.pkl）
        """
        for i, path in enumerate(paths):
            if os.path.exists(path):
                print(f"==> 加载专家 {i+1} 权重: {path}")
                try:
                    state_dict = torch.load(path, map_location='cpu')
                    # 处理不同的权重文件格式
                    if 'model' in state_dict:
                        state_dict = state_dict['model']
                    elif 'state_dict' in state_dict:
                        state_dict = state_dict['state_dict']
                    elif 'model_state_dict' in state_dict:
                        state_dict = state_dict['model_state_dict']
                    
                    # 加载权重
                    self.experts[i].model.load_state_dict(state_dict, strict=False)
                    print(f"==> 专家 {i+1} 权重加载成功")
                except Exception as e:
                    print(f"==> 专家 {i+1} 权重加载失败: {e}")
            else:
                print(f"==> 警告: 专家 {i+1} 权重文件不存在: {path}")

    def forward(self, x, use_gumbel_softmax=True, tau=1.0):
        gate_logits = self.gate.feature_extractor(x)  # [B, N]
        if use_gumbel_softmax:
            # 训练时用Gumbel-Softmax采样one-hot
            gate_weights = torch.nn.functional.gumbel_softmax(gate_logits, tau=tau, hard=True, dim=1)  # [B, N]
        else:
            gate_weights = torch.softmax(gate_logits, dim=1)
        expert_outputs = [expert(x) for expert in self.experts]  # [B, C, H, W] * N
        expert_outputs = torch.stack(expert_outputs, dim=1)      # [B, N, C, H, W]
        gate_weights = gate_weights.view(x.size(0), self.num_experts, 1, 1, 1)
        out = (expert_outputs * gate_weights).sum(dim=1)         # [B, C, H, W]
        return out

    def inference(self, x):
        """
        推理时采用top-1路由，只用权重最大的专家输出。
        """
        gate_weights = self.gate(x)  # [B, N]
        top1_idx = gate_weights.argmax(dim=1)  # [B]
        outputs = []
        for b in range(x.size(0)):
            expert_idx = top1_idx[b].item()
            outputs.append(self.experts[expert_idx](x[b:b+1]))
        return torch.cat(outputs, dim=0)

    def save_all(self, path):
        """
        保存整个MoE模型（包括所有专家和门控网络），建议使用.pkl后缀
        Args:
            path: 保存路径（.pkl）
        """
        torch.save(self.state_dict(), path)

    def load_all(self, path):
        """
        加载整个MoE模型（.pkl文件）
        Args:
            path: 模型文件路径（.pkl）
        """
        if os.path.exists(path):
            state_dict = torch.load(path, map_location='cpu')
            self.load_state_dict(state_dict)
            print(f"==> MoE模型加载成功: {path}")
        else:
            print(f"==> 错误: MoE模型文件不存在: {path}")
    
    def get_expert_usage_stats(self, x):
        """
        获取专家使用统计信息
        
        Args:
            x: 输入数据
            
        Returns:
            usage_stats: 每个专家的使用次数
        """
        gate_weights = self.gate(x)
        top1_idx = gate_weights.argmax(dim=1)
        usage_stats = torch.zeros(self.num_experts)
        for idx in top1_idx:
            usage_stats[idx] += 1
        return usage_stats 

if __name__ == "__main__":
    # 测试MoE模型的参数量和计算量
    import time
    from thop import profile, clever_format
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 定义专家参数 - 只使用一个专家
    expert_args_list = [
        {'mode': 1, 'in_chans': 3, 'out_chans': 3, 'window_size': 16, 'embed_dims': [32, 64, 128, 256, 128, 64, 32], 'depths': [2, 2, 4, 6, 4, 2, 2]}
    ]
    
    # 创建MoE模型
    model = MoEModel(expert_args_list)
    model = model.to(device)
    model.eval()
    
    # 创建测试输入 (1, 3, 256, 256)
    test_input = torch.randn(1, 3, 256, 256).to(device)
    
    print("\n" + "="*50)
    print("MoE模型测试")
    print("="*50)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"冻结参数量: {frozen_params:,}")
    
    # 计算FLOPs
    try:
        flops, params = profile(model, inputs=(test_input,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"计算量 (FLOPs): {flops}")
        print(f"参数量: {params}")
    except Exception as e:
        print(f"FLOPs计算失败: {e}")
        print("请确保已安装thop库: pip install thop")
    
    # 测试推理时间
    print("\n推理时间测试:")
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model(test_input)
        
        # 测试推理时间
        times = []
        for _ in range(100):
            start_time = time.time()
            output = model(test_input)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        print(f"平均推理时间: {avg_time*1000:.2f} ms")
        print(f"FPS: {1/avg_time:.2f}")
    
    # 测试专家使用统计
    print("\n专家使用统计:")
    usage_stats = model.get_expert_usage_stats(test_input)
    for i, usage in enumerate(usage_stats):
        print(f"专家 {i+1}: {usage.item():.0f} 次")
    
    print("\n" + "="*50) 