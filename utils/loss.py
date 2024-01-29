import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .skeletonize import Skeletonize
class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode =='con_ce':
            return self.ConLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, #ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def ConLoss(self, logit, target):
        # loss = torch.mean(torch.sum(-target * torch.log(F.softmax(logit, dim=1)), dim=1))
        # loss = torch.mean(torch.sum(-target * nn.LogSoftmax()(logit), dim=1))
        loss = nn.BCEWithLogitsLoss()(logit, target)
        # loss = nn.BCELoss()(logit, target)
        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.25):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, #ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b

class central_loss(nn.Module):
    def __call__(self, y_pred, y_true):
    
        y_pred[y_pred>0.4]=1
        y_pred[y_pred<.4]=0
        y_pred = y_pred.data.cpu().numpy()
        y_pred=torch.tensor(y_pred).cuda()

        skeletonization_module = Skeletonize(probabilistic=True, beta=0.33, tau=1.0, simple_point_detection='Boolean')

        
        skeleton_stack_1 = torch.zeros_like(y_pred)
        num=1
        for step in range(num):
            skeleton_stack_1 = skeleton_stack_1 + skeletonization_module(y_pred)
        skeleton_1 = (skeleton_stack_1 / num).round()

        
        skeleton_stack_2 = torch.zeros_like(y_pred)
        for step in range(num):
            skeleton_stack_2 = skeleton_stack_2 + skeletonization_module(y_true)
        skeleton_2 = (skeleton_stack_2 / num).round()


        i = torch.sum(skeleton_1)
        j = torch.sum(skeleton_2)
        intersection = torch.sum(skeleton_1 * skeleton_2)
        score = (2. * intersection) / (i + j)
        return 1.0-score

from .soft_skeleton import soft_skel        
def soft_dice(y_true, y_pred):
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return 1. - coeff


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.2, smooth=1.):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_pred,y_true ):
        dice = soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)) + self.smooth) / (
                    torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)) + self.smooth) / (
                    torch.sum(skel_true) + self.smooth)
        cl_dice = 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
        return (1.0 - self.alpha) * dice + self.alpha * cl_dice        

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        smooth = 1e-6
        dice = 0.
        for i in range(pred.size(1)):
            dice += 2 * (pred[:, i] * target[:, i]).sum(dim=1).sum(dim=1) / (
                    pred[:, i].sum(dim=1).sum(dim=1) +
                    target[:, i].sum(dim=1).sum(dim=1) + smooth)
        dice = dice / pred.size(1)
        loss_fun = nn.BCELoss()
        BCE = loss_fun(pred.to(torch.float32), target.to(torch.float32))
        return torch.clamp((1 - dice).mean(), 0, 1) + 4 * BCE

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




