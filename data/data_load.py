import os
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFlip, PairRandomVerticalFlip, PairToTensor, \
    PairRandomScale, PairPatchGenerator, PairRandomRotateFlip
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        """
        Args:
            image_dir: 数据集根目录路径
            transform: 数据增强变换组合 (默认None)
            is_test: 测试模式标志 (默认False)
        """
        self.image_dir = image_dir
        # 获取模糊图像文件名列表
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy/'))
        # 验证文件格式有效性
        self._check_image(self.image_list)
        # 文件排序保证顺序一致性
        self.image_list.sort()
        # 数据增强变换器
        self.transform = transform
        # 测试模式标记（控制返回格式）
        self.is_test = is_test

        # 训练模式预计算样本总数
        if not is_test and transform:
            #self.total_samples = self._calculate_total_samples()
            self.total_samples = len(self.image_list)
        else:
            self.total_samples = len(self.image_list)

    def _calculate_total_samples(self):
        """预计算增强后的总样本数"""
        total = 0
        for img_name in self.image_list:
            img_path = os.path.join(self.image_dir, 'hazy', img_name)
            img = Image.open(img_path).convert('RGB')
            w, h = img.size

            # 计算该图像的分块数量
            patches = 1  # 默认至少1个样本
            for t in self.transform.transforms:
                if isinstance(t, PairPatchGenerator):
                    # 计算分块数量
                    if h < t.patch_size or w < t.patch_size:
                        patches = 1
                    else:
                        patches = ((h - t.patch_size) // t.stride + 1) * \
                                  ((w - t.patch_size) // t.stride + 1)
                    break

            # 计算尺度变换因子（默认为1）
            scales = 1
            for t in self.transform.transforms:
                if isinstance(t, PairRandomScale):
                    scales = len(t.scales)
                    break

            # 计算旋转/翻转因子（默认为1）
            rotations = 1
            for t in self.transform.transforms:
                if isinstance(t, PairRandomRotateFlip):
                    rotations = 8  # PairRandomRotateFlip有8种变换
                    break

            total += (patches * scales * rotations)#扩展循环训练

        return total

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 测试模式直接返回原始图像
        if self.is_test:
            return self._get_test_item(idx % len(self.image_list))

        # 训练模式计算原始图像索引
        orig_idx = idx % len(self.image_list)
        return self._get_train_item(orig_idx)

    def _get_test_item(self, idx):
        """处理测试模式样本加载"""
        img_path = os.path.join(self.image_dir, 'hazy', self.image_list[idx])
        gt_path = os.path.join(self.image_dir, 'gt', self.image_list[idx])
        image = Image.open(img_path).convert('RGB')
        label = Image.open(gt_path).convert('RGB')
        image = F.to_tensor(image)
        label = F.to_tensor(label)
        return image, label, self.image_list[idx]

    def _get_train_item(self, idx):
        """处理训练模式样本加载"""
        img_path = os.path.join(self.image_dir, 'hazy', self.image_list[idx])
        gt_path = os.path.join(self.image_dir, 'gt', self.image_list[idx])
        image = Image.open(img_path).convert('RGB')
        label = Image.open(gt_path).convert('RGB')

        # 应用数据增强变换
        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)

        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError(f"Invalid image format: {x}")


def train_dataloader(path, batch_size=2, num_workers=0, use_transform=True):
    """
    Args:
        path: 数据集根目录路径
        batch_size: 每批样本数 (默认4)
        num_workers: 数据加载线程数 (默认0)
        use_transform: 是否启用数据增强 (默认True)
    """
    image_dir = os.path.join(path, 'train')  # 构建训练集路径

    # 初始化数据增强组合
    transform = None
    if use_transform:
        # 创建同步变换组合：随机裁剪 -> 随机翻转 -> 转张量
        transform = PairCompose(
            [
                #PairRandomScale(),
                PairRandomCrop(128),  # 其余数据集用的是512
                PairPatchGenerator(128), #室外数据集分辨率不够，切块128
                PairRandomRotateFlip(),
                PairRandomHorizontalFlip(),
                PairRandomVerticalFlip(),
                PairToTensor()
            ]
        )

    # 创建DataLoader实例
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform),  # 应用数据增强的数据集
        batch_size=batch_size,
        shuffle=True,  # 打乱训练数据顺序
        num_workers=num_workers,
        pin_memory=True  # 启用内存锁页加速GPU传输
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    """
    Args:
        path: 数据集根目录路径
        batch_size: 每批样本数 (默认1，测试时通常为1)
        num_workers: 数据加载线程数 (默认0)

    Returns:
        DataLoader: 配置好的测试数据加载器
    """
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test'), is_test=True),  # 使用测试集路径并启用测试模式
        batch_size=batch_size,
        shuffle=False,  # 保持数据原始顺序
        num_workers=num_workers,
        pin_memory=True  # 启用内存锁页优化
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    """
    Args:
        path: 数据集根目录路径
        batch_size: 每批样本数 (默认1)
        num_workers: 数据加载线程数 (默认0)

    Returns:
        DataLoader: 配置好的验证数据加载器
    """
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'val'), is_test=True, transform=None),
        batch_size=batch_size,
        shuffle=False,  # 保持数据原始顺序
        num_workers=num_workers  # 单线程加载保证稳定性
    )

    return dataloader