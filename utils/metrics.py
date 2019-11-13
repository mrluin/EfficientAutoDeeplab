import numpy as np


class Evaluator(object):
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes
        self.confusion_matrix = np.zeros((self.nb_classes,) * 2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum()+1e-10)
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1)+1e-10)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix) + 1e-10
        )
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / (np.sum(self.confusion_matrix)+1e-10)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) - np.diag(self.confusion_matrix)+1e-10
        )
        # weighted sum
        FWIoU = (freq[freq>0] * iu[freq>0]).sum()
        return FWIoU

    def Precision(self):
        true_positive = np.diag(self.confusion_matrix) # TP
        predicted_condition_positive = np.sum(self.confusion_matrix, axis=0) # TP+FP
        P_per_class = true_positive / (predicted_condition_positive+1e-10) # TP / (TP+FP)
        P = np.nanmean(P_per_class)
        return P, P_per_class

    def Recall(self):
        true_positive = np.diag(self.confusion_matrix) # TP
        condition_positive = np.sum(self.confusion_matrix, axis=1) # TP+FN
        R_per_class = true_positive / (condition_positive+1e-10) # TP/P
        R = np.nanmean(R_per_class)
        return R, R_per_class

    def Fx_Score(self, x=1):
        _, R_per_class = self.Recall()
        _, P_per_class = self.Precision()
        F = ((x * x + 1) * P_per_class * R_per_class) / (x * x * P_per_class + R_per_class + 1e-10)
        F = np.nanmean(F)
        return F

    def generate_confusion_matrix(self, gt_images, pred_images):
        # 0-dim: means predicted condition
        # 1-dim: means True condition
        mask = (gt_images >= 0) & (gt_images < self.nb_classes)
        label = self.nb_classes * gt_images[mask].astype('int') + pred_images[mask]
        count = np.bincount(label, minlength=self.nb_classes ** 2)
        confusion_matrix = count.reshape(self.nb_classes, self.nb_classes)
        return confusion_matrix

    def add_batch(self, gt_images, logits):
        # here receive cuda.tensor
        #assert gt_images.shape == logits.shape
        gt_images = gt_images.data.cpu().numpy()
        logits = logits.data.cpu().numpy()
        pred_images = np.argmax(logits, axis=1)
        self.confusion_matrix += self.generate_confusion_matrix(gt_images, pred_images)

    def reset_confusion_matrix(self):
        self.confusion_matrix = np.zeros((self.nb_classes,) * 2)


