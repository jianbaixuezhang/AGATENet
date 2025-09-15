import torch
from models.moe import MoEModel
from data import test_dataloader
import torch.nn.functional as F
import os
from skimage.metrics import peak_signal_noise_ratio
from pytorch_msssim import ssim
from torchvision.transforms import functional as TF
from PIL import Image, ImageDraw, ImageFont
from utils import Adder, Timer, load_excel, tensor_metric

def test_moe_model(moe_model, args):
    if args.test_model:
        if os.path.exists(args.test_model):
            print(f"==> 加载测试模型 {args.test_model}")
            checkpoint = torch.load(args.test_model, map_location='cpu')
            
            # 加载模型状态
            if 'model' in checkpoint:
                moe_model.load_state_dict(checkpoint['model'])
            else:
                moe_model.load_state_dict(checkpoint)
        else:
            print(f"警告: 测试模型文件 {args.test_model} 不存在（应为.pkl后缀）")
            return
    
    moe_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    moe_model = moe_model.to(device)

    # 数据加载 - 与原始eval.py保持一致
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)

    # 推理与评估
    psnr_list = []
    ssim_list = []
    
    # 创建结果目录
    result_dir = args.result_dir if hasattr(args, 'result_dir') else 'test_results'
    os.makedirs(result_dir, exist_ok=True)
    
    # 结果记录文件
    result_file = os.path.join(result_dir, 'test_result.txt')
    with open(result_file, 'w') as f_txt:
        f_txt.write("Image Name | PSNR | SSIM\n")
        f_txt.write("--------------------------\n")
    
    with torch.no_grad():
        for idx, (input_img, label_img, name) in enumerate(dataloader):
            input_img = input_img.to(device)
            label_img = label_img.to(device)
            h, w = input_img.shape[-2], input_img.shape[-1]
            pred = moe_model.inference(input_img)
            pred = pred[:, :, :h, :w]
            pred_clip = torch.clamp(pred, 0, 1)
            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()
            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            ssim_val = tensor_metric(label_img, pred_clip, 'SSIM', data_range=1)
            ssim_val = ssim_val.item()
            psnr_list.append(psnr)
            ssim_list.append(ssim_val)
            print(f"{name[0]}: PSNR={psnr:.2f} SSIM={ssim_val:.4f}")
            
            # 记录到文件
            with open(result_file, 'a') as f_txt:
                record = f"{name[0]:<20} {psnr:.2f} dB  {ssim_val:.4f}\n"
                f_txt.write(record)
            
            # 最终测试：生成详细的三联图对比
            if args.save_image:
                # 保存单张结果图
                save_name = os.path.join(result_dir, name[0])
                pred_pil = TF.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred_pil.save(save_name)

    
    # 计算并保存平均结果
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)
    
    with open(result_file, 'a') as f_txt:
        f_txt.write("\n--------------------------------\n")
        f_txt.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f_txt.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f_txt.write(f"Test Images: {len(psnr_list)}\n")
    
    print(f"\n=== 测试结果 ===")
    print(f"平均PSNR: {avg_psnr:.2f} dB")
    print(f"平均SSIM: {avg_ssim:.4f}")
    print(f"测试图像数量: {len(psnr_list)}")
    print(f"结果已保存到: {result_dir}")

