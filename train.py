import os, time, argparse, cv2, math
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import cv2

from data import train_dataloader
from warmup_scheduler import GradualWarmupScheduler
from valid import _valid
from torch.utils.tensorboard import SummaryWriter
from models.former import build_net
from torchvision.models import vgg16
from torchvision import models
from utils import Adder, Timer, load_excel, tensor_metric
from loss import MSSSIM


def _train(model, args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    last_safe_state = {
        'model': None,
        'optimizer': None,
        'scheduler': None,
        'batch': 0
    }

    metric = [['PSNR', 'SSIM']]
    msssim_loss_fn = MSSSIM()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05,
        eps=1e-8
    )

    print('> Loading dataset...')
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)
    max_iter = len(dataloader)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=args.num_epoch,
        steps_per_epoch=len(dataloader),
        pct_start=0.4,
        anneal_strategy='cos',
        div_factor=15.0,
        final_div_factor=1e3
    )

    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        print('Resume from %d' % start_epoch)

    epoch_pixel_adder = Adder()
    iter_pixel_adder = Adder()
    epoch_timer = Timer('m')
    iter_timer = Timer('m')
    best_psnr = -1

    for epoch_idx in range(start_epoch, args.num_epoch + 1):
        epoch_timer.tic()
        model.train()

        for iter_idx, batch_data in enumerate(dataloader):
            iter_timer.tic()
            input_img, GT = batch_data
            input_img = input_img.to(device)
            GT = GT.to(device)

            optimizer.zero_grad()
            pred_img = model(input_img)
            pred_img = torch.clamp(pred_img, 0.0, 1.0)
            GT = torch.clamp(GT, 0.0, 1.0)

            try:
                l1 = F.smooth_l1_loss(pred_img, GT)
                msssim_loss_ = msssim_loss_fn(pred_img, GT)
                beta = 0.01 if epoch_idx < 100 else 0.4
                loss = l1 + beta * msssim_loss_

                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError("Invalid loss value")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()

            except Exception as e:
                print(f"Epoch {epoch_idx} Iter {iter_idx} Error: {str(e)}")
                if last_safe_state['model'] is not None:
                    model.load_state_dict(last_safe_state['model'])
                    optimizer.load_state_dict(last_safe_state['optimizer'])
                    scheduler.load_state_dict(last_safe_state['scheduler'])
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.7
                    print(f"Recovered to batch {last_safe_state['batch']}, lr: {param_group['lr']:.2e}")
                continue

            if (iter_idx + 1) % 10 == 0:
                last_safe_state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'batch': iter}
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch_idx
                }, os.path.join(args.model_save_dir, 'model.pkl'))

            epoch_pixel_adder(l1.item())
            iter_pixel_adder(l1.item())

            if (iter_idx + 1) % args.print_freq == 0:
                mse = tensor_metric(GT, pred_img, 'MSE', data_range=1)
                psnr = tensor_metric(GT, pred_img, 'PSNR', data_range=1)
                ssim = tensor_metric(GT, pred_img, 'SSIM', data_range=1)

                # 严格保持原始打印格式
                print(
                    "Time: %7.4f Epoch: %03d  Iter: %4d/%4d Loss content: %7.6f total loss: %.6f MSE: %.6f PSNR: %.4f SSIM: %.4f" % (
                        iter_timer.toc(), epoch_idx, iter_idx + 1, max_iter,
                        iter_pixel_adder.average(), loss.item(), mse, psnr, ssim))

                iter_pixel_adder.reset()
                iter_timer.tic()

        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch_idx
        }, os.path.join(args.model_save_dir, 'model.pkl'))

        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict()}, save_name)

        print(f"EPOCH: {epoch_idx:02d} Elapsed time: {epoch_timer.toc():4.2f} min")

        print("EPOCH: %02d\nElapsed time: %4.2f Epoch Pixel Loss: %7.4f " % (
            epoch_idx, epoch_timer.toc(), epoch_pixel_adder.average()))
        epoch_pixel_adder.reset()

        if epoch_idx % args.valid_freq == 0:
            val_psnr, val_ssim = _valid(model, args, epoch_idx)
            print('<=======================Epoch %d ===================================>' % epoch_idx)
            print('Epoch %d: Average PSNR %.4f dB; Average SSIM %.4f' % (epoch_idx, val_psnr, val_ssim))
            metric.append([val_psnr, val_ssim])
            load_excel(metric)

            if (val_psnr + val_ssim) > best_psnr:
                best_psnr = (val_psnr + val_ssim)
                torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Best.pkl'))

    torch.save({'model': model.state_dict()}, os.path.join(args.model_save_dir, 'Final.pkl'))