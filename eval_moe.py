import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from utils import Adder
from data import valid_dataloader
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as f
import torchvision.utils


def eval_moe_model(moe_model, args, epoch=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    moe_model.eval()
    factor = 8


    if epoch is not None:
        result_dir = os.path.join(args.result_dir, str(epoch))
    else:
        result_dir = args.result_dir
    
    os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()

        print(f'\nStart Epoch {epoch} Evaluation' if epoch is not None else '\nStart Evaluation')
        total = len(dataloader)

        for iter_idx, data in enumerate(dataloader):
            if len(data) == 3:
                input_img, label_img, img_names = data
                filename = os.path.splitext(os.path.basename(img_names[0]))[0] if img_names else f"{iter_idx:04d}"
            else:
                input_img, label_img = data
                filename = f"{iter_idx:04d}"

            input_img = input_img.to(device)
            label_img = label_img.to(device)

            # 图像尺寸处理 - 与valid.py保持一致
            h, w = input_img.shape[-2], input_img.shape[-1]
            H = ((h + factor) // factor) * factor
            W = ((w + factor) // factor) * factor
            padh = H - h
            padw = W - w

            # 填充处理
            input_img_padded = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            # MoE模型推理
            pred = moe_model.inference(input_img_padded)
            pred = pred[:, :, :h, :w]  # 移除填充
            
            # 数值处理
            pred_clip = torch.clamp(pred, 0, 1)
            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()

            # 计算指标 - 与valid.py保持一致
            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            ssim_val = ssim(pred_numpy, label_numpy, channel_axis=0, data_range=1)

            # 记录指标
            psnr_adder(psnr)
            ssim_adder(ssim_val)

            if args.save_image:
                save_path = os.path.join(result_dir, f"{filename}_pred.png")
                torchvision.utils.save_image(pred_clip, save_path)

            progress = (iter_idx + 1) / total * 100
            print(f'\rProcess: {iter_idx + 1}/{total} [{progress:.1f}%]', end='')

        avg_psnr = psnr_adder.average()
        avg_ssim = ssim_adder.average()
        with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f_txt:
            f_txt.write(f"Epoch: {epoch}\nPSNR: {avg_psnr:.4f}\nSSIM: {avg_ssim:.4f}")

        print(f'\nEpoch {epoch} Validation Complete!' if epoch is not None else '\nValidation Complete!')
        moe_model.train()
        return avg_psnr, avg_ssim

