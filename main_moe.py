import os
import torch
import argparse
from torch.backends import cudnn
from models.moe import MoEModel
from train_moe import train_moe_model
from test_moe import test_moe_model

def main(args):
    cudnn.benchmark = True
    
    # 创建必要的目录
    if not os.path.exists('results/'):
        os.makedirs('results/')
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        # 训练模式：加载预训练的专家模型，训练门控网络
        expert_args_list = [
            ((1,), {'in_chans': 3, 'out_chans': 3, 'window_size': 8}),
            ((1,), {'in_chans': 3, 'out_chans': 3, 'window_size': 8}),
            ((1,), {'in_chans': 3, 'out_chans': 3, 'window_size': 8}),
            ((1,), {'in_chans': 3, 'out_chans': 3, 'window_size': 8}),
        ]
        
        # 构建专家预训练权重路径
        pretrained_paths = [
            os.path.join(args.expert_dir, 'expert_model1.pkl'),
            os.path.join(args.expert_dir, 'expert_model2.pkl'),
            os.path.join(args.expert_dir, 'expert_model3.pkl'),
            os.path.join(args.expert_dir, 'expert_model4.pkl'),
        ]
        
        # 检查专家模型文件是否存在
        for path in pretrained_paths:
            if not os.path.exists(path):
                print(f"错误: 专家模型文件不存在: {path}")
                return
        
        # 构建MoE模型（用于训练门控网络）
        moe_model = MoEModel(expert_args_list, pretrained_paths)
        
        if torch.cuda.is_available():
            moe_model.cuda()
        
        train_moe_model(moe_model, args)
        
    elif args.mode == 'test':
        # 测试模式：加载完整的MoE权重文件
        test_model_path = args.test_model
        if not os.path.isabs(test_model_path):
            # 如果是相对路径，假设在模型保存目录下
            test_model_path = os.path.join(args.model_save_dir, test_model_path)
        
        if not os.path.exists(test_model_path):
            print(f"错误: 测试模型文件不存在: {test_model_path}")
            return
        
        # 从权重文件推断专家数量
        checkpoint = torch.load(test_model_path, map_location='cpu')
        
        # 加载模型状态
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 从权重文件中推断专家数量
        expert_count = 0
        for key in state_dict.keys():
            if key.startswith('experts.'):
                expert_num = int(key.split('.')[1])
                expert_count = max(expert_count, expert_num + 1)
        
        print(f"==> 从权重文件检测到 {expert_count} 个专家")
        
        # 构建MoE模型（用于测试）
        expert_args_list = [
            ((1,), {'in_chans': 3, 'out_chans': 3, 'window_size': 8}) 
            for _ in range(expert_count)
        ]
        
        moe_model = MoEModel(expert_args_list, pretrained_paths=None)
        moe_model.load_state_dict(state_dict)
        
        if torch.cuda.is_available():
            moe_model.cuda()
        
        test_moe_model(moe_model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MoE模型训练和测试')
    parser.add_argument('--model_name', default='MoEDahazeFormer', type=str, help='模型名称')
    
    # 数据集路径
    parser.add_argument('--data_dir', type=str, default='realwordhaze', help='数据集路径')
    parser.add_argument('--data', type=str, default='realword', help='数据集类型')
    
    # 专家模型路径
    parser.add_argument('--expert_dir', type=str, default='results/SEMHSADahazeFormer', 
                       help='专家模型权重文件目录')
    
    # 训练模式或者测试模式
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str, help='运行模式')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num_epoch', type=int, default=400, help='训练迭代次数')

    parser.add_argument('--print_freq', type=int, default=10, help='打印训练日志的频率')
    parser.add_argument('--num_worker', type=int, default=4, help='多进程加载预处理数据')
    parser.add_argument('--save_freq', type=int, default=10, help='模型保存频率(epoch)')
    parser.add_argument('--valid_freq', type=int, default=10, help='验证频率(epoch)')
    parser.add_argument('--resume', type=str, default='', help='恢复训练的检查点路径')
    
    # 测试参数
    parser.add_argument('--test_model', type=str, default='Best.pkl', help='测试模型路径')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False], help='是否保存测试图像')

    args = parser.parse_args()
    
    # 设置模型保存和结果目录
    args.model_save_dir = os.path.join('results/', args.model_name, 'Training-Results/')
    args.result_dir = os.path.join('results/', args.model_name, 'images')

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    if args.mode == 'train':
        # 断点续炼逻辑
        checkpoint_path = os.path.join(args.model_save_dir, 'model.pkl')
        if os.path.exists(checkpoint_path):
            choice = input(f"检测到断点文件 {checkpoint_path}，是否继续训练？(y/n): ")
            if choice.lower().strip() == 'y':
                args.resume = checkpoint_path
                print("==> 恢复训练")
            else:
                args.resume = ''
                print("==> 重新开始训练")
        else:
            print("==> 未找到断点文件，开始新的训练")

    main(args) 