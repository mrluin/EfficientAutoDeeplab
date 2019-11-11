import os
import sys
import torch
import logging
import random
import time
import numpy as np

def print_experiment_environment():

    print('\n','-'*30+'->\tworking with PyTorch version {}'.format(torch.__version__))
    print('-'*30+'->\twith cuda version {}'.format(torch.version.cuda))
    print('-'*30+'->\tcudnn enabled: {}'.format(torch.backends.cudnn.enabled))
    print('-'*30+'->\tcudnn version: {}'.format(torch.backends.cudnn.version()))

    print('-'*30+'->\tcudnn benchmark: {}'.format(torch.backends.cudnn.benchmark))
    print('-'*30+'->\tcudnn deterministic: {}'.format(torch.backends.cudnn.deterministic))

def set_manual_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_logger(args, log_file_name):

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save_path, log_file_name))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

class AverageMeter(object):
    """
        # Computes and stores the average and current value
    """
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    @property
    def value(self):
        return self.val

    @property
    def average(self):
        return self.avg

    @property
    def _get_sum(self):
        return self.sum


def get_monitor_metric(monitor_metric, loss, acc, miou, fscore):
    if monitor_metric == 'loss':
        return loss
    elif monitor_metric == 'acc':
        return acc
    elif monitor_metric == 'miou':
        return miou
    elif monitor_metric == 'fscore':
        return fscore
    else:
        raise ValueError('do not support monitor_metric:{}'.format(monitor_metric))

def time_for_file():
    ISOTIMEFORMAT = '%d-%h-at-%H-%M-%S'
    return '{}'.format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))

def get_padding_size(kernel_size, dilation):

    return int((kernel_size-1)/2*dilation)

def delta_ij(i, j):
    if i == j:
        return 1
    else:
        return 0

def detach_variable(inputs):
    if isinstance(inputs, tuple):
        return tuple([detach_variable(x) for x in inputs])
    else:
        x = inputs.detach()
        x.requires_grad = inputs.requires_grad
        return x

def count_conv_flop(layer, x):
    # dilations do not change conv_flops
    out_h = int(x.size()[2] / layer.stride[0])
    out_w = int(x.size()[3] / layer.stride[1])
    delta_ops = layer.inc * layer.outc * layer.kernel_size * layer.kernel_size * \
        out_h * out_w / layer.groups

    return delta_ops

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def save_inter_tensor(list, val):
    if len(list) <= 2:
        list.append(val)
        return list
    else:
        list.pop(0)
        list.append(val)
        return list