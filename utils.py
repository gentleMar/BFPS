import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

# pranet
class StructureLoss(torch.nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, target):
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        pred = torch.sigmoid(pred)
        inter = ((pred * target) * weit).sum(dim=(2, 3))
        union = ((pred + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()


from medpy.metric import dc,jc,hd95,sensitivity,specificity


def dice_coefficient(pred, gt):
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()

    return dc(pred_np, gt_np)

def iou_coefficient(pred, gt):
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()

    return jc(pred_np, gt_np)


# these might be helpful when using other medical image segmentation datasets
def accuracy_coefficient(pred, gt):
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()

    return (pred_np == gt_np).sum() / gt_np.size

def sensitivity_coefficient(pred, gt):
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()

    return sensitivity(pred_np, gt_np)

def specificity_coefficient(pred, gt):
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()

    return specificity(pred_np, gt_np)

# ignore=True to be the same hd95 calculating method with previous works (swinunet, ...)
def hd95_coefficient(pred, gt, ignore=True, default=np.nan):
    pred_np = pred.cpu().numpy()
    gt_np = gt.cpu().numpy()

    if np.sum(gt_np) > 0 and np.sum(pred_np) > 0:
        return hd95(pred_np, gt_np)
    elif ignore:
        return 0
    else:
        return default

def multiclass_dice_coefficient(pred, gt, num_classes, separate_classes=False):
    dices = []
    for i in range(num_classes):
        pred_i = (pred == i)
        gt_i = (gt == i)
        if pred_i.sum() > 0 and gt_i.sum() > 0:
            dice = dice_coefficient(pred_i, gt_i)
        elif pred_i.sum() > 0:
            # to be the same dsc calculating method with previous works (swinunet, ...)
            dice = 1
        else:
            dice = 0
        dices.append(dice)
    if separate_classes:
        return dices
    else:
        return sum(dices) / len(dices)
    
def multiclass_iou_coefficient(pred, gt, num_classes, separate_classes=False):
    ious = []
    for i in range(num_classes):
        pred_i = (pred == i)
        gt_i = (gt == i)
        iou = iou_coefficient(pred_i, gt_i)
        ious.append(iou)
    if separate_classes:
        return ious
    else:
        return sum(ious) / len(ious)
    

def multiclass_hd95_coefficient(pred, gt, num_classes, separate_classes=False):
    hd95s = []
    for i in range(num_classes):
        pred_i = (pred == i)
        gt_i = (gt == i)
        hd95 = hd95_coefficient(pred_i, gt_i)
        hd95s.append(hd95)
    if separate_classes:
        return hd95s
    else:
        return sum(hd95s) / len(hd95s)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass