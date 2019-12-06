import json
import os
import sys
import torch
import logging
import random
import time
import numpy as np
import shutil
import PIL
__all__ = ['print_experiment_environment', 'set_manual_seed', 'set_logger',
           'AverageMeter', 'get_list_index_split', 'get_next_scale', 'get_list_index', 'get_cell_decode_type',
           'get_prev_c', 'get_padding_size', 'get_monitor_metric', 'get_prev_c_abs', 'get_scale_relation',
           'time_for_file', 'detach_variable', 'delta_ij', 'create_exp_dir',
           'count_parameters', 'count_normal_conv_flop', 'count_conv_flop', 'save_inter_tensor',
           ]

def print_experiment_environment():
    info = "Python  Version  : {:}".format(sys.version.replace('\n', ' '))
    info += "\nPillow  Version  : {:}".format(PIL.__version__)
    info += "\nPyTorch Version  : {:}".format(torch.__version__)
    info += "\ncuDNN   Version  : {:}".format(torch.backends.cudnn.version())
    info += "\nCUDA available   : {:}".format(torch.cuda.is_available())
    info += "\nCUDA GPU numbers : {:}".format(torch.cuda.device_count())
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        info += "\nCUDA_VISIBLE_DEVICES={:}".format(os.environ['CUDA_VISIBLE_DEVICES'])
    else:
        info += "\nDoes not set CUDA_VISIBLE_DEVICES"

    return info

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
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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
        #x.requires_grad = inputs.requires_grad
        x.requires_grad = inputs.requires_grad
        return x

def count_conv_flop(layer, x):
    # dilations do not change conv_flops
    if isinstance(layer.stride, tuple):
        out_h = int(x.size()[2] / layer.stride[0])
        out_w = int(x.size()[3] / layer.stride[1])
    else:
        out_h = int(x.size()[2] / layer.stride)
        out_w = int(x.size()[3] / layer.stride)
    if isinstance(layer.kernel_size, tuple):
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * \
            out_h * out_w / layer.groups
    else:
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size * layer.kernel_size * \
                    out_h * out_w / layer.groups

    return delta_ops

def count_normal_conv_flop(layer, x):
    if isinstance(layer.stride, tuple):
        out_h = int(x.size()[2] / layer.stride[0])
        out_w = int(x.size()[3] / layer.stride[1])
    else:
        out_h = int(x.size()[2] / layer.stride)
        out_w = int(x.size()[3] / layer.stride)
    if isinstance(layer.kernel_size, tuple):
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * \
            out_h * out_w / layer.groups
    else:
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size * layer.kernel_size * \
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

def get_next_scale(choice, current_scale):
    # scale in 0, 1, 2, 3, 4
    if choice == 1:
        return current_scale
    elif choice == 0:
        return current_scale -1
    elif choice == 2:
        return current_scale + 1

def get_list_index(layer, scale):
    if layer == 0:
        return scale
    elif layer == 1:
        return 2+scale
    elif layer == 2:
        return 5+scale
    elif layer >= 3:
        return 4*(layer-3)+9+scale

def get_list_index_split(layer, current_scale, next_scale):
    if layer == 0:
        return next_scale - current_scale
    elif layer == 1:
        base = 2
        if current_scale == 0:
            return base + next_scale - current_scale
        elif current_scale == 1:
            if next_scale - current_scale == -1:
                return base + 2 + (current_scale - 1)*3
            elif next_scale - current_scale == 0:
                return base + 2 + (current_scale - 1)*3 + 1
            else:
                return base + 2 + (current_scale - 1)*3 + 2
    elif layer == 2:
        base = 7
        if current_scale == 0:
            return base + (next_scale - current_scale)
        elif current_scale == 1 or current_scale == 2:
            if next_scale - current_scale == -1:
                return base + 2 + (current_scale - 1)*3
            elif next_scale - current_scale == 0:
                return base + 2 + (current_scale - 1)*3 + 1
            elif next_scale - current_scale == 1:
                return base + 2 + (current_scale - 1)*3 + 2
    else:
        base = 15 + (layer - 3) * 10
        if current_scale == 0:
            return base + (next_scale - current_scale)
        elif current_scale == 1 or current_scale == 2:
            if next_scale - current_scale == 0:
                return base + 2 + (current_scale - 1)*3 + 1
            elif next_scale - current_scale == 1:
                return base + 2 + (current_scale - 1)*3 + 2
            elif next_scale - current_scale == -1:
                return base + 2 + (current_scale - 1)*3
        elif current_scale == 3:
            if next_scale - current_scale == 0:
                return base + 2 + (current_scale - 1)*3 + 1
            elif next_scale - current_scale == -1:
                return base + 2 + (current_scale - 1)*3

def get_cell_index(layer, current_scale, next_scale):
    # pay attention:: layer-0 is w.r.t. output of stem2
    # for gumbel_super_network order
    # use layer + 1
    relation2index = {1: 0, 0: 1, -1: 2} # next_scale - current_scale
    relation2index_s0 = {0: 0, -1: 1}
    layer = layer + 1 # real layer_index
    if layer == 1:
        return next_scale
    elif layer == 2:
        base = 2
        if next_scale == 0:
            return base + relation2index_s0[next_scale-current_scale]
        elif next_scale == 1:
            return base + 2 + relation2index[next_scale-current_scale]
        elif next_scale == 2:
            return base + 4 + relation2index[next_scale - current_scale]
        else:
            raise ValueError('Error in get_cell_index, scale 3 cannot appear in layer 2')
    elif layer == 3:
        base = 7
        if next_scale == 0:
            return base + relation2index_s0[next_scale-current_scale]
        elif next_scale == 1:
            return base + 2 + relation2index[next_scale-current_scale]
        elif next_scale == 2:
            return base + 5 + relation2index[next_scale-current_scale]
        elif next_scale == 3:
            return base + 7 + relation2index[next_scale-current_scale]
    else:
        base = 15 + (layer-4) * 10
        if next_scale == 0:
            return base + relation2index_s0[next_scale-current_scale]
        elif next_scale == 1:
            return base + 2 + relation2index[next_scale-current_scale]
        elif next_scale == 2:
            return base + 5 + relation2index[next_scale-current_scale]
        elif next_scale == 3:
            return base + 8 + relation2index[next_scale-current_scale]

def get_prev_c(intermediate_features, scale):
    # scale is next scale
    if intermediate_features[-2][0] == scale:
        return intermediate_features[-2][1], intermediate_features[-1][1]
    else:
        return None, intermediate_features[-1][1]

def get_prev_c_abs(intermediate_features, scale):
    if np.abs(intermediate_features[-2][0] - scale) <= 1:
        return intermediate_features[-2][1], intermediate_features[-1][1]
    else:
        return None, intermediate_features[-1][1]

def get_cell_decode_type(current_scale, next_scale):
    if current_scale == next_scale:
        return 'same'
    elif current_scale == next_scale - 1:
        return 'reduction'
    elif current_scale == next_scale + 1:
        return 'up'
def get_scale_relation(scale, next_scale):

    raise NotImplementedError

def get_pfeatures(layer, scale, scale0_features, scale1_features, scale2_features, scale3_features):

    prev_prev_feature = None
    prev_feature = []
    if scale == 0:
        if layer == 0: prev_prev_feature = None
        else: prev_prev_feature = scale0_features[-2]
        if layer == 0: prev_feature.append(scale0_features[-1])
        else:
            prev_feature.append(scale0_features[-1])
            prev_feature.append(scale1_features[-1])
    elif scale == 1:
        if layer == 0 or layer == 1: prev_prev_feature = None
        else: prev_prev_feature = scale1_features[-2]
        if layer == 0: prev_feature.append(scale0_features[-1])
        elif layer == 1:
            prev_feature.append(scale0_features[-1])
            prev_feature.append(scale1_features[-1])
        else:
            prev_feature.append(scale0_features[-1])
            prev_feature.append(scale1_features[-1])
            prev_feature.append(scale2_features[-1])
    elif scale == 2:
        if layer == 0: raise ValueError('invalid layer scale relation 0, 2')
        elif layer == 1 or layer == 2: prev_prev_feature = None
        else: prev_prev_feature = scale2_features[-2]
        if layer == 1: prev_feature.append(scale1_features[-1])
        elif layer == 2:
            prev_feature.append(scale1_features[-1])
            prev_feature.append(scale2_features[-1])
        else:
            prev_feature.append(scale1_features[-1])
            prev_feature.append(scale2_features[-1])
            prev_feature.append(scale3_features[-1])
    elif scale == 3:
        if layer == 0 or layer == 1: raise ValueError('invalid layer scale relation 0/1, 3')
        elif layer == 2 or layer == 3: prev_prev_feature = None
        else: prev_prev_feature = scale3_features[-2]
        if layer == 2: prev_feature.append(scale2_features[-1])
        else:
            prev_feature.append(scale2_features[-1])
            prev_feature.append(scale3_features[-1])
    else: raise ValueError('invalid scale value {}'.format(scale))

    return prev_prev_feature, prev_feature
'''
def network_layer_to_space(net_arch, nb_layers):
    assert len(net_arch) == nb_layers, 'invalid nb_layers'
    network_space = np.zeros((nb_layers, 4, 3))

    # record scale from 1 to 12
    # i            from 0 to 11
    # in layer i in which scale, what the choice is
    prev = 0
    for i, scale in enumerate(net_arch):
        if scale > i + 1:
            raise ValueError('invalid scale {} in layer {}'.format(scale, i+1))
        if scale == prev:
            rate = 1
        elif scale == prev+1:
            rate = 2
        elif scale == prev-1:
            rate = 0
        else:
            raise ValueError('invalid scale and prev_scale relation')

        network_space[i][0][rate] = 1
        prev = scale

    return network_space
'''
def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir: {}'.format(path))

    if scripts_to_save  is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def detect_none_inputs(s0, s1):
    log = ''
    if s0 is None:
        log += 'None, '
    else:
        log += 'not None, '
    if s1 is None:
        log += 'None'
    else:
        log += 'not None'

    return log

def detect_inputs_shape(s0, s1):
    log = ''
    if s0 is None:
        log += 's0: None,'
    else:
        log += 's0: {:}'.format(s0.shape)

    if s1 is None:
        log += ' s1: None'
    else:
        log += ' s1: {:}'.format(s1.shape)

    return log

def append_scale_list(scale_list, scale0_features, scale1_features, scale2_features, scale3_features):
    for key in scale_list.keys():
        if scale_list[key] is not None:
            if key == 0:
                scale0_features.append(scale_list[key])
            elif key == 1:
                scale1_features.append(scale_list[key])
            elif key == 2:
                scale2_features.append(scale_list[key])
            elif key == 3:
                scale3_features.append(scale_list[key])
            else:
                raise ValueError('invalid key error {}'.format(key))
        else:
            continue


def detect_invalid_index(index, nb_layers):
    # index shape as [12, 4]
    for i in range(0, nb_layers):
        if i == 0:
            if index[i][0] != 1: return 'layer {} scale 0 invalid_value {}'.format(i+1, index[i][0]), False
            if index[i][1] != 0: return 'layer {} scale 1 invalid_value {}'.format(i+1, index[i][1]),False
        elif i == 1:
            if index[i][0] != 1 and index[i][0] != 2: return 'layer {} scale 0 invalid_value {}'.format(i+1, index[i][0]), False
            if index[i][1] != 0 and index[i][1] != 1: return 'layer {} scale 1 invalid_value {}'.format(i+1, index[i][1]),False
            if index[i][2] != 0: return 'layer {} scale 2 invalid_value {}'.format(i, index[i][2]),False
        elif i == 2:
            if index[i][0] != 1 and index[i][0] != 2: return 'layer {} scale 0 invalid_value {}'.format(i+1, index[i][0]),False
            if index[i][2] != 0 and index[i][2] != 1: return 'layer {} scale 2 invalid_value {}'.format(i+1, index[i][2]),False
            if index[i][3] != 0: return 'layer {} scale 3 invalid_value {}'.format(i, index[i][3]),False
        else:
            if index[i][0] != 1 and index[i][0] != 2: return 'layer {} scale 0 invalid_value {}'.format(i+1, index[i][0]),False
            if index[i][3] != 0 and index[i][3] != 1: return 'layer {} scale 3 invalid_value {}'.format(i+1, index[i][3]),False

    return None, True


def save_configs(configs, save_path, phase):

    if configs is not None:
        config_path = os.path.join(save_path, '{:}.config'.format(phase))
        print('=' * 30 + '\n' + 'Run Configs dumps to {}'.format(config_path))
        json.dump(configs, open(config_path, 'w'), indent=4)
    else:
        raise ValueError('configs is None, cannot save')

def convert_secs2time(epoch_time, return_str=False):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600*need_hour) / 60)
    need_secs = int(epoch_time - 3600*need_hour - 60*need_mins)
    if return_str:
        str = '[{:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        return str
    else:
        return need_hour, need_mins, need_secs

def configs_resume(args, configs_dict, phase):
    if phase == 'search':
        for key in args.keys():
            if 'path' not in key and 'gpu_ids' not in key and 'resume' not in key and 'resume_file' not in key:
                args.__dict__[key] = configs_dict[key]
        return args
    elif phase == 'retrain':
        for key in args.keys():
            if 'path' not in key and 'gpu_ids' not in key and'resume_from_retrain' not in key and 'resume_file' not in key and 'checkpoint_file' not in key:
                args.__dict__[key] = configs_dict[key]
        return args
    else: raise ValueError('phase {:} do not supports'.format(phase))