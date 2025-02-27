import torch
import torch.nn.functional as F

def weighted_iou_loss(pred, target):
    """
    Compute the weighted IoU loss.
    pred: predicted saliency map (tensor)
    target: ground truth saliency map (tensor)
    """
    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection
    wiou = 1 - (intersection + 1) / (union + 1)
    return wiou.mean()

def ig_loss(pred_saliency, pred_edge, gt_saliency, gt_edge, m1=4, m2=4):
    """
    Compute the Interactive Guidance Loss (IGLoss) with dynamic balancing.

    pred_saliency: Predicted saliency map (tensor).
    pred_edge: Predicted edge map (tensor).
    gt_saliency: Ground truth saliency map (tensor).
    gt_edge: Ground truth edge map (tensor).
    m1, m2: Balancing factors for easy and hard examples.
    """
    # Edge-guided saliency loss (Ls)
    bce_loss_saliency = F.binary_cross_entropy_with_logits(pred_saliency, gt_saliency, reduction='none')
    saliency_weight = pred_edge.sigmoid() * m1 + 1
    weighted_bce_loss_saliency = (saliency_weight * bce_loss_saliency).mean()

    # Saliency-guided edge loss (Le)
    bce_loss_edge = F.binary_cross_entropy_with_logits(pred_edge, gt_edge, reduction='none')
    edge_weight = (1 - pred_saliency.sigmoid()) * m2 + 1
    weighted_bce_loss_edge = (edge_weight * bce_loss_edge).mean()

    # Total IGLoss
    ig_loss_value = weighted_bce_loss_saliency + weighted_bce_loss_edge
    return ig_loss_value

def total_loss(preds_saliency, preds_edge, gt_saliency, gt_edge, lambdas=(1, 1)):
    """
    Compute the total loss with deep supervision.

    preds_saliency: List of predicted saliency maps from different stages.
    preds_edge: List of predicted edge maps from different stages.
    gt_saliency: Ground truth saliency map (tensor).
    gt_edge: Ground truth edge map (tensor).
    lambdas: Tuple of lambda values (lambda_1, lambda_2) for loss components.
    """
    lambda_1, lambda_2 = lambdas
    losses = []

    # Calculate IGLoss and WIoU for each stage
    for i, (pred_saliency, pred_edge) in enumerate(zip(preds_saliency, preds_edge)):
        ig = ig_loss(pred_saliency, pred_edge, gt_saliency, gt_edge)
        wiou = weighted_iou_loss(pred_saliency.sigmoid(), gt_saliency)

        # Combine losses with weighting
        li = lambda_1 * ig + lambda_2 * wiou

        # Apply stage-specific weighting
        stage_weight = 1 / (2 ** max(0, i - 1))
        losses.append(stage_weight * li)

    return sum(losses)

if __name__ == "__main__":
    # 示例使用
    # 模拟模型输出
    smap1, smap2, smap3, smap4, smap5 = [torch.rand(4, 1, 448, 448) for _ in range(5)]  # 5个检测阶段的显著性预测
    edge1, edge2, edge3, edge4, edge5 = [torch.rand(4, 1, 448, 448) for _ in range(5)]  # 5个检测阶段的边缘预测
    gt_saliency = torch.rand(4, 1, 448, 448)  # Ground truth saliency map
    gt_edge = torch.rand(4, 1, 448, 448)      # Ground truth edge map

    # 计算总损失
    loss = total_loss([smap1, smap2, smap3, smap4, smap5], [edge1, edge2, edge3, edge4, edge5], gt_saliency, gt_edge, lambdas=(1, 1))
    print(f"Total Loss: {loss.item()}")
