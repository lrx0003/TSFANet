import numpy as np
import torch
class metrics():

    def __init__(self):
        self.negative_iou = []
        self.postive_iou = []
        self.dice = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.mae = []

    def reset(self):
        self.negative_iou = []
        self.postive_iou = []
        self.dice = []
        self.accuracy = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.mae = []

    def add(self, pred, mask, threshold):
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        # pred = pred.cpu().numpy()
        pred_mask = np.zeros_like(pred)
        pred_mask[pred >= threshold] = 1
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()

        # mask = mask.cpu().numpy()

        pred_num_of_negative = np.sum(pred_mask == 0)  # TN + FN
        pred_num_of_positive = np.sum(pred_mask == 1)  # TP + FP
        mask_num_of_negative = np.sum(mask == 0)  # TN + FP
        mask_num_of_positive = np.sum(mask == 1)  # TP + FN

        TP = np.sum((pred_mask == 1) & (mask == 1))
        FP = pred_num_of_positive - TP
        FN = mask_num_of_positive - TP
        TN = pred_num_of_negative - FN

        smooth = 1e-5
        # IoU
        negative_iou = (TN + smooth) / (TN + FN + FP + smooth)
        postive_iou = (TP + smooth) / (TP + FN + FP + smooth)
        self.negative_iou.append(negative_iou)
        self.postive_iou.append(postive_iou)

        # Dice
        dice = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
        self.dice.append(dice)

        # Accuracy
        accuracy = (TP + TN) / (TN + FN + TP + FP)
        self.accuracy.append(accuracy)

        # Precision
        precision = (TP + smooth) / (TP + FP + smooth)
        self.precision.append(precision)

        # Recall\Sensitivity
        recall = (TP + smooth) / (TP + FN + smooth)
        self.recall.append(recall)

        # F1-score
        f1 = (2 * precision * recall + smooth) / (precision + recall + smooth)
        self.f1.append(f1)

        # Mean absolute error
        mae = round(np.mean(np.abs(pred_mask - mask)).astype(np.float64), 4)
        self.mae.append(mae)

    def add_batch(self, pred, mask, threshold):
        batch = pred.shape[0]
        for index in range(0, batch):
            self.add(pred[index], mask[index], threshold)

    # def show(self):
    #     return 'mIoU:{} mDice:{} accuracy:{} precision:{} recall:{} f1-score:{} MAE:{}'.format(
    #         round(np.mean(self.postive_iou), 4),
    #         round(np.mean(self.dice), 4),
    #         round(np.mean(self.accuracy), 4),
    #         round(np.mean(self.precision), 4),
    #         round(np.mean(self.recall), 4),
    #         round(np.mean(self.f1), 4),
    #         round(np.mean(self.mae), 4),
    #     )
    def show(self):
        return {
            'mIoU': round(np.mean(self.postive_iou), 4) if len(self.postive_iou) > 0 else 0.0,
            'mDice': round(np.mean(self.dice), 4) if len(self.dice) > 0 else 0.0,
            'accuracy': round(np.mean(self.accuracy), 4) if len(self.accuracy) > 0 else 0.0,
            'precision': round(np.mean(self.precision), 4) if len(self.precision) > 0 else 0.0,
            'recall': round(np.mean(self.recall), 4) if len(self.recall) > 0 else 0.0,
            'f1-score': round(np.mean(self.f1), 4) if len(self.f1) > 0 else 0.0,
            'MAE': round(np.mean(self.mae), 4) if len(self.mae) > 0 else 0.0
        }