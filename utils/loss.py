import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self,
                 weight=None,
                 label_smoothing=0.,
                 cuda=0):
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.cuda = cuda

    def calculate_loss(self, logits, targets):
        if self.label_smoothing > 0.:
            return self.CrossEntropyLossWithLabelSmoothing(logits, targets, self.label_smoothing)
        else:
            return self.CrossEntropyLoss(logits, targets)

    def CrossEntropyLoss(self, logits, targets):
        criterion = nn.CrossEntropyLoss(weight=self.weight)
        if self.cuda >= 0:
            criterion = criterion.to('cuda:{}'.format(self.cuda))

        return criterion(logits, targets)

    def CrossEntropyLossWithLabelSmoothing(self, logits, targets, label_smoothing):
        logsoftmax = nn.Softmax()
        nb_classes = logits.size(1)
        targets = torch.unsqueeze(targets, 1)
        soft_targets = torch.zeros_like(logits)
        soft_targets.scatter_(1, targets, 1)
        # label smoothing
        soft_targets = soft_targets * (1 - label_smoothing) + label_smoothing / nb_classes
        return torch.mean(torch.sum(- soft_targets * logsoftmax(logits), 1))

