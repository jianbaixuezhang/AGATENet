import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
import torch
from PIL import Image
import cv2


# class PairRandomRotateFlip(object):
#     def __call__(self, img, gt):
#         mode = np.random.randint(0, 8)
#         if mode == 0: return img, gt
#         if mode == 1: return img.transpose(Image.FLIP_TOP_BOTTOM), gt.transpose(Image.FLIP_TOP_BOTTOM)
#         if mode == 2: return img.rotate(90), gt.rotate(90)
#         if mode == 3: return img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM), gt.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
#         if mode == 4: return img.rotate(180), gt.rotate(180)
#         if mode == 5: return img.rotate(180).transpose(Image.FLIP_TOP_BOTTOM), gt.rotate(180).transpose(Image.FLIP_TOP_BOTTOM)
#         if mode == 6: return img.rotate(270), gt.rotate(270)
#         return img.rotate(270).transpose(Image.FLIP_TOP_BOTTOM), gt.rotate(270).transpose(Image.FLIP_TOP_BOTTOM)
class PairRandomRotateFlip(object):
    def __call__(self, img, gt):
        mode = np.random.randint(0, 8)
        if mode == 0: return img, gt
        if mode == 1: return img.transpose(Image.FLIP_TOP_BOTTOM), gt.transpose(Image.FLIP_TOP_BOTTOM)
        if mode == 2: return img.rotate(90), gt.rotate(90)
        if mode == 3: return img.rotate(90).transpose(Image.FLIP_TOP_BOTTOM), gt.rotate(90).transpose(Image.FLIP_TOP_BOTTOM)
        if mode == 4: return img.rotate(180), gt.rotate(180)
        if mode == 5: return img.rotate(180).transpose(Image.FLIP_TOP_BOTTOM), gt.rotate(180).transpose(Image.FLIP_TOP_BOTTOM)
        if mode == 6: return img.rotate(270), gt.rotate(270)
        return img.rotate(270).transpose(Image.FLIP_TOP_BOTTOM), gt.rotate(270).transpose(Image.FLIP_TOP_BOTTOM)

#随机裁剪
# class PairRandomCrop(transforms.RandomCrop):
#     def __call__(self, image, GT):
#         # 确保图像为 PIL.Image 或 Tensor 格式
#         width, height = image.size  # PIL 格式的 size 是 (width, height)
#
#
#         # 高度填充（若需要）
#         if self.pad_if_needed and height < self.size[0]:
#             pad_height = self.size[0] - height
#             # 使用上下对称填充
#             image = F.pad(image, (0, pad_height // 2, 0, pad_height - pad_height // 2), self.fill, self.padding_mode)
#             GT = F.pad(GT, (0, pad_height // 2, 0, pad_height - pad_height // 2), self.fill, self.padding_mode)
#
#         # 宽度填充（若需要）
#         if self.pad_if_needed and width < self.size[1]:
#             pad_width = self.size[1] - width
#             # 使用左右对称填充
#             image = F.pad(image, (pad_width // 2, 0, pad_width - pad_width // 2, 0), self.fill, self.padding_mode)
#             GT = F.pad(GT, (pad_width // 2, 0, pad_width - pad_width // 2, 0), self.fill, self.padding_mode)
#
#         # 执行随机裁剪
#         i, j, h, w = self.get_params(image, self.size)
#         return F.crop(image, i, j, h, w), F.crop(GT, i, j, h, w)

class PairRandomCrop(transforms.RandomCrop):
    def __call__(self, image, GT):
        width, height = image.size
        padding_mode = 'reflect'  # 反射填充

        # 高度填充
        if self.pad_if_needed and height < self.size[0]:
            pad_height = self.size[0] - height
            image = F.pad(image, (0, pad_height//2, 0, pad_height - pad_height//2),
                         padding_mode=padding_mode)
            GT = F.pad(GT, (0, pad_height//2, 0, pad_height - pad_height//2),
                       padding_mode=padding_mode)

        # 宽度填充
        if self.pad_if_needed and width < self.size[1]:
            pad_width = self.size[1] - width
            image = F.pad(image, (pad_width//2, 0, pad_width - pad_width//2, 0),
                         padding_mode=padding_mode)
            GT = F.pad(GT, (pad_width//2, 0, pad_width - pad_width//2, 0),
                       padding_mode=padding_mode)

        i, j, h, w = self.get_params(image, self.size)
        return F.crop(image, i, j, h, w), F.crop(GT, i, j, h, w)

#匹配变换
class PairCompose(transforms.Compose):
    def __call__(self, image, GT):
        # 遍历组合中的每个数据增强变换
        for t in self.transforms:
            # 同步应用相同变换到图像和标签对
            image, GT = t(image, GT)
        # 返回处理后的图像标签对
        return image, GT

#同步随机水平翻转图像
class PairRandomHorizontalFlip(object):
    def __call__(self, img, GT):
        """
        Args:
            img: 原始图像(PIL Image)
            GT: 对应标签(PIL Image)
        Returns:
            tuple: 随机翻转后的图像标签对，保持空间一致性
        """
        if random.random() < 0.5:
            # 以概率p执行水平翻转，保持图像和标签的同步变换【0.5】
            return F.hflip(img), F.hflip(GT)
        # 保持原始图像标签对不变
        return img, GT

#同步随机垂直翻转图像
class PairRandomVerticalFlip(object):
    def __call__(self, img, GT):
        """
        Args:
            img: 原始图像(PIL Image)
           GT: 对应标签(PIL Image)

        Returns:
            tuple: 随机翻转后的图像标签对，保持空间一致性
        """
        if random.random() < 0.5:
            # 以概率p执行水平翻转，保持图像和标签的同步变换
            return  F.vflip(img), F.vflip(GT)
        # 保持原始图像标签对不变
        return img, GT


class PairToTensor(transforms.ToTensor):
    """
       同步将图像-标签对转换为张量
       Args:
           pic: 输入图像(PIL Image或numpy数组)
           GT: 对应标签(PIL Image或numpy数组)

       Returns:
           tuple: (图像张量, 标签张量) 形状为(C,H,W)的浮点型张量
     """
    def __call__(self, pic, GT):
        img_tensor = F.to_tensor(pic)
        gt_tensor = F.to_tensor(GT)
        # NaN检查
        if torch.isnan(img_tensor).any() or torch.isnan(gt_tensor).any():
            raise ValueError("NaN detected in data augmentation!")
        return img_tensor, gt_tensor

#多尺度随机缩放
class PairRandomScale(object):
    def __init__(self, scales=[0.8, 1.0, 1.2]):
        self.scales = scales

    def __call__(self, img, gt):
        scale = random.choice(self.scales)
        w, h = img.size
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, Image.BICUBIC), gt.resize(new_size, Image.BICUBIC)

#动态分块处理器
# class PairPatchGenerator(object):
#     def __init__(self, patch_size=256, stride=128):
#         self.patch_size = patch_size
#         self.stride = stride
#
#     def __call__(self, img, gt):
#         img = np.array(img)
#         gt = np.array(gt)
#
#         h, w = img.shape[:2]
#         patches = []
#         for y in range(0, h - self.patch_size + 1, self.stride):
#             for x in range(0, w - self.patch_size + 1, self.stride):
#                 # 处理边界情况
#                 y_end = min(y + self.patch_size, h)
#                 x_end = min(x + self.patch_size, w)
#
#                 patch_img = img[y:y_end, x:x_end] if y_end <= h else img[h - self.patch_size:h, x:x_end]
#                 patch_gt = gt[y:y_end, x:x_end] if y_end <= h else gt[h - self.patch_size:h, x:x_end]
#
#                 # 填充不足部分
#                 if patch_img.shape[:2] != (self.patch_size, self.patch_size):
#                     patch_img = cv2.resize(patch_img, (self.patch_size, self.patch_size))
#                     patch_gt = cv2.resize(patch_gt, (self.patch_size, self.patch_size))
#
#                 patches.append((
#                     Image.fromarray(patch_img),
#                     Image.fromarray(patch_gt)
#                 ))
#         return random.choice(patches)

class PairPatchGenerator(object):
    def __init__(self, patch_size=256, stride=128):
        self.patch_size = patch_size
        self.stride = stride

    def __call__(self, img, gt):
        img = np.array(img)
        gt = np.array(gt)
        h, w = img.shape[:2]

        # 尺寸不足时返回原图
        if h < self.patch_size or w < self.patch_size:
            return [(Image.fromarray(img), Image.fromarray(gt))]

        patches = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                y_end = y + self.patch_size
                x_end = x + self.patch_size
                patch_img = img[y:y_end, x:x_end]
                patch_gt = gt[y:y_end, x:x_end]
                patches.append((Image.fromarray(patch_img), Image.fromarray(patch_gt)))
        return random.choice(patches)