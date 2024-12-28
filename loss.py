# loss_functions.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# class BinaryFocalLoss(nn.Module):
#     """
#     Binary Focal Loss for binary classification tasks.
#     Reference:
#     https://arxiv.org/pdf/1708.02002.pdf
#     """

#     def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
#         """
#         Initializes the BinaryFocalLoss module.

#         Args:
#             alpha (float, optional): Weighting factor for the positive class. Default is 0.25.
#             gamma (float, optional): Focusing parameter for modulating factor (1-pt)^gamma. Default is 2.
#             reduction (str, optional): Specifies the reduction to apply to the output:
#                                        'none' | 'mean' | 'sum'. Default is 'mean'.
#         """
#         super(BinaryFocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         if reduction not in ['none', 'mean', 'sum']:
#             raise ValueError("Reduction must be one of 'none', 'mean', or 'sum'")
#         self.reduction = reduction

#     def forward(self, inputs, targets):
#         """
#         Forward pass for BinaryFocalLoss.

#         Args:
#             inputs (torch.Tensor): Predicted logits with shape (N, 1, H, W).
#             targets (torch.Tensor): Ground truth masks with shape (N, 1, H, W).

#         Returns:
#             torch.Tensor: Computed focal loss.
#         """
#         # 确保 inputs 和 targets 形状一致
#         if inputs.shape != targets.shape:
#             raise ValueError(f"Input shape {inputs.shape} and target shape {targets.shape} must be the same.")

#         # Apply sigmoid to get probabilities
#         probs = torch.sigmoid(inputs)
#         probs = probs.view(-1)
#         targets = targets.view(-1)

#         # Calculate BCE loss
#         BCE_loss = F.binary_cross_entropy_with_logits(inputs.view(-1), targets, reduction='none')

#         # Calculate pt
#         pt = torch.where(targets == 1, probs, 1 - probs)
#         pt = pt.clamp(min=1e-8, max=1 - 1e-8)  # To avoid log(0)

#         # Calculate focal loss
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for binary classification tasks.
    Reference:
    https://arxiv.org/pdf/1708.02002.pdf
    Modified to handle alpha as a scalar for positive class and (1 - alpha) for negative class.
    """

    def __init__(self, alpha=0.25, gamma=2, reduction='mean', smooth=1e-5):
        """
        Initializes the BinaryFocalLoss module.

        Args:
            alpha (float, optional): Weighting factor for the positive class. Default is 0.25.
            gamma (float, optional): Focusing parameter for modulating factor (1-pt)^gamma. Default is 2.
            reduction (str, optional): Specifies the reduction to apply to the output:
                                       'none' | 'mean' | 'sum'. Default is 'mean'.
            smooth (float, optional): Smoothing factor to avoid log(0). Default is 1e-5.
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Reduction must be one of 'none', 'mean', or 'sum'")
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Forward pass for BinaryFocalLoss.

        Args:
            inputs (torch.Tensor): Predicted logits with shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth masks with shape (N, 1, H, W).

        Returns:
            torch.Tensor: Computed focal loss.
        """
        # 确保 inputs 和 targets 形状一致
        if inputs.shape != targets.shape:
            raise ValueError(f"Input shape {inputs.shape} and target shape {targets.shape} must be the same.")

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Calculate BCE loss with logits for numerical stability
        BCE_loss = F.binary_cross_entropy_with_logits(inputs.view(-1), targets, reduction='none')

        # Calculate pt
        pt = torch.where(targets == 1, probs, 1 - probs)
        pt = pt.clamp(min=self.smooth, max=1.0 - self.smooth)  # To avoid log(0)

        # Calculate alpha_t
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Calculate focal loss
        focal_loss = alpha_t * torch.pow((1 - pt), self.gamma) * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation tasks.
    """

    def __init__(self, smooth=1.0):
        """
        Initializes the DiceLoss module.

        Args:
            smooth (float, optional): Smoothing factor to avoid division by zero. Default is 1.0.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        """
        Forward pass for DiceLoss.

        Args:
            prediction (torch.Tensor): Predicted probabilities with shape (N, 1, H, W).
            target (torch.Tensor): Ground truth masks with shape (N, 1, H, W).

        Returns:
            torch.Tensor: Computed dice loss.
        """
        # Flatten tensors
        prediction = prediction.view(-1)
        target = target.view(-1)

        # Calculate intersection and sums
        intersection = (prediction * target).sum()
        sum_pred = prediction.sum()
        sum_target = target.sum()

        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (sum_pred + sum_target + self.smooth)

        # Return Dice loss
        return 1 - dice

class CombinedLoss(nn.Module):
    """
    Combined Loss function that integrates BCEWithLogitsLoss, DiceLoss, and BinaryFocalLoss with specified weights.
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.3, focal_weight=0.2, alpha=0.25, gamma=2, reduction='mean', smooth=1e-5):
        """
        Initializes the CombinedLoss module.

        Args:
            bce_weight (float, optional): Weight for BCE loss. Default is 0.5.
            dice_weight (float, optional): Weight for Dice loss. Default is 0.3.
            focal_weight (float, optional): Weight for Focal loss. Default is 0.2.
            alpha (float, optional): Alpha parameter for BinaryFocalLoss. Default is 0.25.
            gamma (float, optional): Gamma parameter for BinaryFocalLoss. Default is 2.
            reduction (str, optional): Reduction method for BinaryFocalLoss. Default is 'mean'.
            smooth (float, optional): Smoothing factor for DiceLoss. Default is 1e-5.
        """
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.focal = BinaryFocalLoss(alpha=alpha, gamma=gamma, reduction=reduction)

    def forward(self, logits, targets):
        """
        Forward pass for CombinedLoss.

        Args:
            logits (torch.Tensor): Predicted logits with shape (N, 1, H, W).
            targets (torch.Tensor): Ground truth masks with shape (N, 1, H, W).

        Returns:
            torch.Tensor: Computed combined loss.
        """
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(torch.sigmoid(logits), targets)
        focal_loss = self.focal(logits, targets)

        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss + self.focal_weight * focal_loss
        return total_loss

#############################################
# 新增：structure_loss 函数
#############################################


def structure_loss(pred, mask):
    """
    结构化损失函数，融合了加权 BCE 与加权 IoU 的思想:
    
    1) wbce：加权的 BCE。通过 “weit” 来突出某些像素在误差中的权重（例如边缘区域）。
       wbce = (weit * bce_loss).sum() / weit.sum()
    
    2) wiou：加权 IoU。通过在交并集中加入 “weit”，更加关注重点像素区域。
       wiou = 1 - (inter + 1)/(union - inter + 1)
    
    两者相加得到本次分割的结构化损失。
    
    pred: (B, 1, H, W) 模型原始 logits 输出
    mask: (B, 1, H, W) Ground Truth 掩码 (0/1)
    """
    # 计算权重：weit
    # 用 avg_pool2d() 求出局部平均，再和原 mask 做差，
    # 其绝对值越大表示不确定区域权重越高。
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    # --------------- 加权 BCE ---------------
    # BCE with logits，不做 reduce，得到每个像素的 loss
    wbce_map = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    # 加权后求平均
    wbce = (weit * wbce_map).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    # --------------- 加权 IoU ---------------
    # 先将 logits -> 概率
    pred_sig = torch.sigmoid(pred)
    inter = ((pred_sig * mask) * weit).sum(dim=(2, 3))
    union = ((pred_sig + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    # --------------- 总和 ---------------
    return (wbce + wiou).mean()