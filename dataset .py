# datasets/dataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import random
from torchvision.transforms import InterpolationMode  # 确保导入 InterpolationMode

class JointTransform:
    def __init__(self, resize=(352, 352), horizontal_flip=True, vertical_flip=True, normalize=True, rotation=False, color_jitter=False):
        """
        初始化联合变换
        Args:
            resize (tuple): 图像和掩码调整的尺寸
            horizontal_flip (bool): 是否应用随机水平翻转
            vertical_flip (bool): 是否应用随机垂直翻转
            normalize (bool): 是否对图像进行归一化
            rotation (bool): 是否应用随机旋转
            color_jitter (bool): 是否应用颜色抖动
        """
        self.resize = resize
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.normalize = normalize
        self.rotation = rotation
        self.color_jitter = color_jitter

        self.image_resize = transforms.Resize(self.resize, interpolation=transforms.InterpolationMode.BILINEAR)
        self.mask_resize = transforms.Resize(self.resize, interpolation=transforms.InterpolationMode.NEAREST)
        
        self.image_to_tensor = transforms.ToTensor()
        self.mask_to_tensor = transforms.ToTensor()

        if self.normalize:
            self.normalize_transform = transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
        else:
            self.normalize_transform = None

        if self.rotation:
            self.rotation_transform = transforms.RandomRotation(10)  # 随机旋转±10度
        else:
            self.rotation_transform = None

        if self.color_jitter:
            self.color_jitter_transform = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        else:
            self.color_jitter_transform = None

    def __call__(self, image, mask):
        # 随机水平翻转
        if self.horizontal_flip and random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        # 随机垂直翻转
        if self.vertical_flip and random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)

        # 随机旋转
        if self.rotation_transform:
            angle = random.uniform(-10, 10)
            image = transforms.functional.rotate(image, angle, interpolation=InterpolationMode.BILINEAR)
            mask = transforms.functional.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        # 随机颜色抖动
        if self.color_jitter_transform:
            image = self.color_jitter_transform(image)

        # 调整大小
        image = self.image_resize(image)
        mask = self.mask_resize(mask)

        # 转换为Tensor
        image = self.image_to_tensor(image)
        mask = self.mask_to_tensor(mask)

        # 归一化图像
        if self.normalize_transform:
            image = self.normalize_transform(image)

        return image, mask


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        自定义分割数据集
        Args:
            images_dir (str): 图像文件夹路径
            masks_dir (str): 掩码文件夹路径
            transform (callable, optional): 联合变换
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        assert len(self.images) == len(self.masks), "图像和掩码数量不匹配"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 加载图像和掩码
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 假设掩码为单通道

        # 应用联合变换
        if self.transform:
            image, mask = self.transform(image, mask)

        # 将掩码转换为二值标签（假设前景为1，背景为0）
        mask = torch.where(mask > 0.5, 1, 0).float()

        return image, mask


class SegmentationDataset1(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        """
        自定义分割数据集
        Args:
            images_dir (str): 图像文件夹路径
            masks_dir (str): 掩码文件夹路径
            transform (callable, optional): 联合变换
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        # 获取图像文件列表
        self.images = sorted(os.listdir(images_dir))
        self.masks_map = {}

        # 根据掩码命名规则建立一个映射
        for mask_file in os.listdir(masks_dir):
            # 提取掩码的基名，例如 "Misc_1_pixels0.png" -> "Misc_1"
            base_name = mask_file.split("_pixels0")[0]
            self.masks_map[base_name] = mask_file

        # 仅保留有对应掩码的图像
        self.image_mask_pairs = []
        for img_file in self.images:
            base_name = os.path.splitext(img_file)[0]  # 去掉扩展名，例如 "Misc_1.png" -> "Misc_1"
            if base_name in self.masks_map:
                self.image_mask_pairs.append((img_file, self.masks_map[base_name]))

        assert len(self.image_mask_pairs) > 0, "没有找到匹配的图像和掩码文件"

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        # 获取图像和掩码文件名
        img_file, mask_file = self.image_mask_pairs[idx]

        # 构造文件路径
        img_path = os.path.join(self.images_dir, img_file)
        mask_path = os.path.join(self.masks_dir, mask_file)

        # 加载图像和掩码
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 假设掩码为单通道

        # 应用联合变换
        if self.transform:
            image, mask = self.transform(image, mask)

        # 将掩码转换为二值标签（假设前景为1，背景为0）
        mask = torch.where(mask > 0.5, 1, 0).float()

        return image, mask

