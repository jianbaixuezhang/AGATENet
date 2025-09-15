import time
import os
import pandas as pd
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
import torch

#加法累加器
class Adder(object):
    def __init__(self):
        self.count = 0
        self.num = float(0)

    def reset(self):
        self.count = 0
        self.num = float(0)

    def __call__(self, num):
        self.count += 1
        self.num += num

    def average(self):
        return self.num / self.count


class Timer(object):
    def __init__(self, option='s'):
        self.tm = 0
        self.option = option
        if option == 's':
            self.devider = 1
        elif option == 'm':
            self.devider = 60
        else:
            self.devider = 3600

    def tic(self):
        self.tm = time.time()

    def toc(self):
        return (time.time() - self.tm) / self.devider


def check_lr(optimizer):
    """获取优化器的当前学习率
    注意：返回最后一个参数组的学习率，当存在多个参数组时可能需要调整
    """
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_group['lr']  # 遍历所有参数组，最后保留最后一个的lr
    return lr

#性能指标计算
def tensor_metric(img, imclean, model, data_range=1):  # 计算图像PSNR输入为Tensor

    img_cpu = img.data.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
    imgclean = imclean.data.cpu().numpy().astype(np.float32).transpose(0, 2, 3, 1)
    SUM = 0
    for i in range(img_cpu.shape[0]):

        if model == 'PSNR':
            SUM += compare_psnr(imgclean[i, :, :, :], img_cpu[i, :, :, :], data_range=data_range)
        elif model == 'MSE':
            SUM += compare_mse(imgclean[i, :, :, :], img_cpu[i, :, :, :])
        elif model == 'SSIM':
            SUM += compare_ssim(imgclean[i, :, :, :], img_cpu[i, :, :, :],channel_axis=2, data_range=data_range)
        else:
            print('Model False!')
    return SUM / img_cpu.shape[0]


def load_excel(metric):
    df = pd.DataFrame(metric[1:], columns=metric[0])  # 第一行为列名
    # 使用 with 语句自动保存
    with pd.ExcelWriter('train_log_HN23.xlsx', engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='metrics')


def save_checkpoint(state, checkpoint, name, epoch=0, psnr=0, ssim=0, i = None):#保存学习率
    if i is None:
        torch.save(state, checkpoint + name + '_%d_%.4f_%.4f.tar'%(epoch, psnr, ssim))
    else:
        torch.save(state, checkpoint + name + '_%d_%d_%.4f_%.4f.tar'%(epoch, i, psnr, ssim))