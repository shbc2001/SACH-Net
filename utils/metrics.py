import numpy as np


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Pixel_Precision(self):
        self.precision = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[0, 1])
        return self.precision

    def Pixel_Recall(self):
        self.recall = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 1] + self.confusion_matrix[1, 0])
        return self.recall

    def Pixel_F1(self):
        f1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        return f1

    def Intersection_over_Union(self):
        IoU = np.zeros(shape=(len(self.confusion_matrixs)), dtype=np.float32)
        for i, confusion_matrix in enumerate(self.confusion_matrixs):
            IoU[i] = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0] + confusion_matrix[0, 1] + 1e-10)
        return IoU.mean()

    def Mean_Intersection_over_Union(self):
        MIoU = np.zeros(shape=(len(self.confusion_matrixs)), dtype=np.float32)
        for i, confusion_matrix in enumerate(self.confusion_matrixs):
            M = np.diag(confusion_matrix) / (
                        np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) -
                        np.diag(confusion_matrix))
            MIoU[i] = np.nanmean(M)
        return MIoU.mean()

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Dice_coefficient(self):
        # 获取混淆矩阵的维度
        # 初始化变量来存储TP、FP和FN
        dice = np.zeros(shape=(len(self.confusion_matrixs)), dtype=np.float32)
        for i , confusion_matrix in enumerate(self.confusion_matrixs):
            TP = confusion_matrix[1, 1]
            FP = confusion_matrix[1, 0]
            FN = confusion_matrix[0, 1]
            dice[i] = (2 * TP) / (2 * TP + FP + FN)
        return dice.mean()
    # def Dice_coefficient(self):
    #     num_class = self.confusion_matrix.shape[0]
    #     dice_scores = np.zeros(num_class)
    #
    #     for i in range(num_class):
    #         tp = self.confusion_matrix[i, i]
    #         fp = np.sum(confusion_matrix[i, :]) - tp
    #         fn = np.sum(confusion_matrix[:, i]) - tp
    #         dice_scores[i] = (2 * tp) / (2 * tp + fp + fn)
    #         break
    #
    #     return dice_scores[0]
        

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask].astype('int')
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrixs.append(self._generate_matrix(gt_image, pre_image))

    def reset(self):
        self.confusion_matrixs = []




