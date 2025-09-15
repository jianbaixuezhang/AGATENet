import os
import torch
import argparse
from torch.backends import cudnn
from models.former import build_net
from eval import _eval
from train import _train

def main(args):
    cudnn.benchmark = True
    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    mode = [args.mode, args.data]
    model = build_net(args.data)
    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)
    elif args.mode == 'test':
        _eval(model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='AGATENet', type=str)
    parser.add_argument('--data_dir', type=str, default='dehaze')
    parser.add_argument('--data', type=str, default='NH23')
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=800)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--valid_freq', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='runs', help='TensorBoard log directory')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--warmup_epochs', type=int, default=10)

    parser.add_argument('--test_model', type=str, default='results/AGATENet/Training-Results/Best.pkl')
    parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])
    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, args.data, 'Training-Results/')
    args.result_dir = os.path.join('results/', args.model_name, 'images', args.data)

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    if args.mode == 'train':
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



