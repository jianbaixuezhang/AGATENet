import torch
from data import valid_dataloader
from utils import Adder
import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import torchvision.utils
from pathvalidate import sanitize_filename
import torch.nn.functional as F


def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_set = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    factor = 8
    psnr_adder = Adder()
    ssim_adder = Adder()

    # 创建结果目录
    result_dir = os.path.join(args.result_dir, str(ep))
    os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        print(f'\nStart Epoch {ep} Evaluation')
        total = len(data_set)

        for idx, data in enumerate(data_set):
            # 数据解析
            if len(data) == 3:
                input_img, label_img, img_names = data
                filename = os.path.splitext(os.path.basename(img_names[0]))[0]
            else:
                input_img, label_img = data
                filename = f"{idx:04d}"

            # 文件名清洗
            filename = sanitize_filename(filename)

            # 设备转移
            input_img = input_img.to(device)
            h, w = input_img.shape[2], input_img.shape[3]

            # 统一使用整图处理（按需填充到8的倍数）
            if h % factor == 0 and w % factor == 0:
                pred = model(input_img)
            else:
                H = ((h + factor) // factor) * factor
                W = ((w + factor) // factor) * factor
                input_padded = F.pad(input_img, pad=(0, W - w, 0, H - h), mode='reflect')
                pred = model(input_padded)[:, :, :h, :w]

            # 后续处理
            pred_clip = torch.clamp(pred, 0, 1)

            # 保存与指标计算
            save_path = os.path.join(result_dir, f"{filename}_pred.png")
            torchvision.utils.save_image(pred_clip, save_path)

            p_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)
            SSIM = ssim(p_numpy, label_numpy, channel_axis=0, data_range=1)

            psnr_adder(psnr)
            ssim_adder(SSIM)

            # 进度显示
            progress = (idx + 1) / total * 100
            print(f'\rProcess: {idx + 1}/{total} [{progress:.1f}%]', end='')

        # 保存指标
        with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
            f.write(f"Epoch: {ep}\nPSNR: {psnr_adder.average():.4f}\nSSIM: {ssim_adder.average():.4f}")

    print(f'\nEpoch {ep} Validation Complete!')
    model.train()
    return psnr_adder.average(), ssim_adder.average()