"""
test_model.py

用于测试训练好的分割模型，评估性能指标并可视化预测结果。
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import SegmentationDataset, JointTransform,SegmentationDataset1  # 假设这些模块已实现
from utils.metrics import compute_metrics,compute_metrics_efficient  # 假设您的 metrics 模块已实现
from utils.helpers import load_checkpoint
from SAM2UNet import SAM2UNet  # 假设模型类已实现
import torch.optim as optim


# -----------------------------
# 1. 可视化函数
# -----------------------------
def visualize_results(images, masks, preds, save_dir, batch_idx):
    """
    可视化测试结果并保存到指定文件夹。
    Args:
        images (Tensor): 输入图像 (N, C, H, W)
        masks (Tensor): 真实掩码 (N, 1, H, W)
        preds (Tensor): 预测掩码 (N, 1, H, W)
        save_dir (str): 保存结果的文件夹路径
        batch_idx (int): 当前批次索引
    """
    os.makedirs(save_dir, exist_ok=True)

    for i in range(images.size(0)):
        # 取单张图像、真实掩码和预测掩码
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + 
                      np.array([0.485, 0.456, 0.406]), 0, 1)
        mask = masks[i].cpu().numpy().squeeze()
        pred = preds[i].cpu().numpy().squeeze()

        # 可视化
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("input")
        plt.imshow(img)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.title("truth")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.title("pred")
        plt.imshow(pred, cmap='gray')
        plt.axis('off')

        # 保存结果
        plt.savefig(os.path.join(save_dir, f"batch_{batch_idx}_sample_{i}.png"))
        plt.close()


# -----------------------------
# 2. 测试函数
# -----------------------------
def test_model(model, dataloader, device, save_dir="test_results"):
    """
    测试训练好的模型并评估性能。
    Args:
        model (nn.Module): 训练好的模型
        dataloader (DataLoader): 测试数据加载器
        device (torch.device): 设备 (CPU 或 GPU)
        save_dir (str): 可视化结果的保存路径
    """
    model.eval()  # 设置模型为评估模式
    preds = []
    truths = []
    pbar = tqdm(dataloader, desc="测试中")
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            # 前向传播
            outputs, _, _ = model(images)

            # 将预测结果转换为概率并阈值化
            probs = torch.sigmoid(outputs)
            preds_batch = (probs > 0.5).float()

            # 收集预测值和真实值
            preds.append(preds_batch.view(-1).cpu().numpy())
            truths.append(masks.view(-1).cpu().numpy())

            # 可视化当前批次结果
            visualize_results(images, masks, preds_batch, save_dir, batch_idx)

    # 将所有批次的预测值和真实值连接起来
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)

    # 计算性能指标
    metrics = compute_metrics_efficient(preds, truths)
    print("\n测试结果：")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"F1 分数: {metrics['f1_score']:.4f}")
    print(f"IoU: {metrics['iou']:.4f}")

    return metrics


# -----------------------------
# 3. 主函数
# -----------------------------
def main():
    # ==========================
    # 配置参数
    # ==========================
    #TEST_IMAGES_DIR = "data/test/MDvsFA/image"
    #TEST_MASKS_DIR = "data/test/MDvsFA/mask"
    TEST_IMAGES_DIR = "data/test/SIRST/image"
    TEST_MASKS_DIR = "data/test/SIRST/mask"
    MODEL_PATH = "best_model8.pth"  # 训练好的模型权重
    SAVE_DIR = "test_results-sirst"  # 可视化结果的保存路径

    # ==========================
    # 数据增强和预处理
    # ==========================
    joint_transform = JointTransform(
        resize=(352, 352),
        horizontal_flip=False,
        vertical_flip=False,
        normalize=True
    )

    # ==========================
    # 创建测试数据集和数据加载器
    # ==========================
    test_dataset = SegmentationDataset1(TEST_IMAGES_DIR, TEST_MASKS_DIR, transform=joint_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=8)

    # ==========================
    # 加载模型
    # ==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SAM2UNet().to(device)
    LEARNING_RATE = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    epoch, step, best_f1 = load_checkpoint(MODEL_PATH, model, optimizer, device)

    print(f"加载完成：Epoch: {epoch}, Step: {step}, Best F1: {best_f1:.4f}")

    print(f"加载模型: {MODEL_PATH}")

    # ==========================
    # 测试模型
    # ==========================
    test_metrics = test_model(model, test_loader, device, SAVE_DIR)
    print("\n最终测试指标:")
    print(test_metrics)


if __name__ == "__main__":
    main()
