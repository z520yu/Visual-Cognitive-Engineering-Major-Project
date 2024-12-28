# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np  # 确保已导入numpy
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # 导入 TensorBoard

from SAM2UNet import SAM2UNet  # 确保路径正确
from dataset import SegmentationDataset, JointTransform,SegmentationDataset1
from utils.metrics import compute_metrics,compute_metrics_efficient
from utils.helpers import save_checkpoint
from loss import structure_loss
from loss import CombinedLoss  # 导入 CombinedLoss

def validate(model, val_loader, device):
    model.eval()
    val_loss = 0
    preds = []
    truths = []
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc="验证中", leave=False)
        for images, masks in pbar_val:
            images = images.to(device)
            masks = masks.to(device)

            # 多尺度输出：假设 model(images) -> (pred0, pred1, pred2)
            # 如果实际只有一个输出，请根据你的模型结构调整
            pred0, pred1, pred2 = model(images)

            # 分别计算三个输出的损失并加和
            loss0 = structure_loss(pred0, masks)
            loss1 = structure_loss(pred1, masks)
            loss2 = structure_loss(pred2, masks)
            loss = loss0 + loss1 + loss2

            val_loss += loss.item()

            # 取其中一个尺度 (比如 pred0) 进行评估/可视化
            probs = torch.sigmoid(pred0)
            preds_batch = (probs > 0.5).float()
            # preds_batch = (probs > 0.5).cpu().numpy().astype(int)
            # truths_batch = masks.cpu().numpy().astype(int)
            # 收集预测值和真实值
            preds.append(preds_batch.view(-1).cpu().numpy())
            truths.append(masks.view(-1).cpu().numpy())

            # preds.extend(preds_batch.flatten())
            # truths.extend(truths_batch.flatten())
    # 将所有批次的预测值和真实值连接起来
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    avg_val_loss = val_loss / len(val_loader)
    metrics = compute_metrics_efficient(preds, truths)
    return avg_val_loss, metrics['f1_score'], metrics['iou']


def main():
    # ==========================
    # 配置参数
    # ==========================
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 16
    SAVE_FREQUENCY = 50  # 每 300 步验证一次并保存最佳模型
    LOG_DIR = "runs/sam2_unet9"  # TensorBoard日志目录

    # 数据集路径（请根据实际情况修改）
    train_images_dir = "data/train/image"
    train_masks_dir = "data/train/mask"
    val_images_dir = "data/test/MDvsFA/image"
    val_masks_dir = "data/test/MDvsFA/mask"
    # val_images_dir = "data/test/SIRST/image"
    # val_masks_dir = "data/test/SIRST/mask"

    # 预训练模型路径（如果有）
    model_checkpoint = "pretrained/sam2_hiera_large.pt"  # 例如 "pretrained/sam2_checkpoint.pth"

    # ==========================
    # 数据增强和预处理
    # ==========================
    joint_transform_train = JointTransform(
        resize=(352, 352),
        horizontal_flip=True,
        vertical_flip=True,
        normalize=True,
        rotation=True,      # 添加随机旋转
        color_jitter=True   # 添加颜色抖动
    )

    joint_transform_val = JointTransform(
        resize=(352, 352),
        horizontal_flip=False,
        vertical_flip=False,
        normalize=True
    )

    # ==========================
    # 创建数据集和数据加载器
    # ==========================
    train_dataset = SegmentationDataset(train_images_dir, train_masks_dir, transform=joint_transform_train)
    val_dataset = SegmentationDataset(val_images_dir, val_masks_dir, transform=joint_transform_val)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    # ==========================
    # 设备配置
    # ==========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # ==========================
    # 模型、损失函数、优化器、调度器
    # ==========================
    model = SAM2UNet(checkpoint_path=model_checkpoint).to(device)
    
    # 定义损失函数
    criterion = CombinedLoss(
        bce_weight=0.3, 
        dice_weight=0.3, 
        focal_weight=0.4, 
        alpha=0.75,          # BinaryFocalLoss 的 alpha
        gamma=2,             # BinaryFocalLoss 的 gamma
        reduction='mean',    # BinaryFocalLoss 的 reduction
        smooth=1e-5          # DiceLoss 的 smooth
    )

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # 调度器 (StepLR 只是示例，可根据需要更换)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    # ==========================
    # TensorBoard设置
    # ==========================
    writer = SummaryWriter(log_dir=LOG_DIR)

    # ==========================
    # 训练循环
    # ==========================
    best_f1 = 0.0
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")

        # 训练阶段
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc="训练中", leave=False)

        for step, (images, masks) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)

            # 前向传播：返回多尺度输出 pred0, pred1, pred2
            pred0, pred1, pred2 = model(images)

            # 计算多尺度损失
            loss0 = criterion(pred0, masks)
            loss1 = criterion(pred1, masks)
            loss2 = criterion(pred2, masks)
            loss = loss0 + loss1 + loss2

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 更新进度条的后缀信息
            pbar.set_postfix({'当前损失': f"{loss.item():.4f}"})

            # 记录训练损失到 TensorBoard
            writer.add_scalar('Loss/train', loss.item(), global_step)

            # 按照 SAVE_FREQUENCY 步进行一次验证
            if (step + 1) % SAVE_FREQUENCY == 0:
                val_loss, f1_score, miou = validate(model, val_loader, device)
                
                # 记录验证损失和指标到 TensorBoard
                writer.add_scalar('Loss/validation', val_loss, global_step)
                writer.add_scalar('F1/validation', f1_score, global_step)
                writer.add_scalar('IoU/validation', miou, global_step)

                # 如果当前的 F1 分数更好，则保存模型
                if f1_score > best_f1:
                    best_f1 = f1_score
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_f1': best_f1
                    }, filename="best_model9.pth")
                    print(f"保存最佳模型 at step {global_step} with F1分数: {best_f1:.4f}")

                model.train()

            global_step += 1
        # 打印 epoch 平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        print(f"训练损失: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/train_avg', avg_train_loss, epoch + 1)
        
        val_loss, f1_score, miou = validate(model, val_loader, device)
                
        # 记录验证损失和指标到 TensorBoard
        writer.add_scalar('Loss/validation', val_loss, global_step)
        writer.add_scalar('F1/validation', f1_score, global_step)
        writer.add_scalar('IoU/validation', miou, global_step)

        # 如果当前的 F1 分数更好，则保存模型
        if f1_score > best_f1:
            best_f1 = f1_score
            save_checkpoint({
                'epoch': epoch + 1,
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1
            }, filename="best_model7.pth")
            print(f"保存最佳模型 at step {global_step} with F1分数: {best_f1:.4f}")

        model.train()

        # 更新学习率
        scheduler.step()

    # ==========================
    # 最终验证（可选）
    # ==========================
    model.eval()
    val_loss = 0
    preds = []
    truths = []
    with torch.no_grad():
        pbar_val = tqdm(val_loader, desc="最终验证中", leave=False)
        for images, masks in pbar_val:
            images, masks = images.to(device), masks.to(device)

            # 多尺度输出
            pred0, pred1, pred2 = model(images)
            loss0 = structure_loss(pred0, masks)
            loss1 = structure_loss(pred1, masks)
            loss2 = structure_loss(pred2, masks)
            loss = loss0 + loss1 + loss2

            val_loss += loss.item()

            # 以 pred0 为主要评估输出
            probs = torch.sigmoid(pred0)
            preds_batch = (probs > 0.5).cpu().numpy().astype(int)
            truths_batch = masks.cpu().numpy().astype(int)

            preds.extend(preds_batch.flatten())
            truths.extend(truths_batch.flatten())

    avg_val_loss = val_loss / len(val_loader)
    metrics = compute_metrics(preds, truths)
    print(f"最终验证损失: {avg_val_loss:.4f} | 准确率: {metrics['accuracy']:.4f} | "
          f"F1 分数: {metrics['f1_score']:.4f} | IoU: {metrics['iou']:.4f}")

    # 记录最终验证损失和指标到 TensorBoard
    writer.add_scalar('Final Validation/Loss', avg_val_loss, NUM_EPOCHS)
    writer.add_scalar('Final Validation/Accuracy', metrics['accuracy'], NUM_EPOCHS)
    writer.add_scalar('Final Validation/F1_Score', metrics['f1_score'], NUM_EPOCHS)
    writer.add_scalar('Final Validation/mIoU', metrics['iou'], NUM_EPOCHS)

    # 若最终 F1 仍然高于现有 best_f1，则可再次保存
    if metrics['f1_score'] > best_f1:
        best_f1 = metrics['f1_score']
        save_checkpoint({
            'epoch': NUM_EPOCHS,
            'step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1
        }, filename="best_model_final.pth")
        print("最终最佳模型已保存！")

    writer.close()


if __name__ == "__main__":
    main()
