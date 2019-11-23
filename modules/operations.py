import torch.nn.functional as F
import torch

from modules.my_modules import *
from collections import OrderedDict
from utils.common import get_padding_size
from utils.common import count_conv_flop
from utils.common import count_normal_conv_flop

def set_layer_from_config(layer_config):

    if layer_config is None:
        return None
    name2layer = {
        ConvLayer.__name__: ConvLayer,
        DilConv.__name__: DilConv,
        SepConv.__name__: SepConv,
        Identity.__name__: Identity,
        Zero.__name__: Zero,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
        FactorizedReduce.__name__: FactorizedReduce,
        FactorizedIncrease.__name__: FactorizedIncrease
    }
    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return  layer.build_from_config(layer_config)

def build_activation(act_func, inplace=False):
    if act_func == 'relu':
        return nn.ReLU(inplace=inplace)
    elif act_func == 'relu6':
        return nn.ReLU6(inplace=inplace)
    elif act_func == 'tanh':
        return nn.Tanh()
    elif act_func == 'sigmoid':
        return nn.Sigmoid()
    elif act_func == None:
        return None
    else:
        raise ValueError('do not support {}'.format(act_func))



# TODO should we consider BN affine?

# nn.Module --> MyModule --> My2DLayer
class My2DLayer(MyModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bn=True,
                 act_func='relu',
                 dropout_rate=0.,
                 ops_order='act_weight_bn'):

        super(My2DLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        modules = {}
        # use 'act_weight_bn' by default
        if self.use_bn:
            modules['bn'] = nn.BatchNorm2d(self.out_channels)
        else:
            modules['bn'] = None

        # activation
        modules['act'] = build_activation(self.act_func, inplace=False)
        # dropout
        if self.dropout_rate > 0.:
            modules['dropout'] = nn.Dropout2d(self.dropout_rate, inplace=True)
        else:
            modules['dropout'] = None
        # weight
        modules['weight'] = self.weight_op()

        # add modules
        # ops_list act, weight, bn
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'act':
                self.add_module('act', modules['act'])
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    def weight_op(self):
        raise  NotImplementedError

    def forward(self, x):
        # self._modules is OrderedDict self.children == self._modules
        for module in self._modules.values():
            x = module(x)
        return x
    @property
    def module_str(self):
        raise NotImplementedError
    @property
    def config(self):
        return {
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'use_bn': self.use_bn,
            'act_func': self.act_func,
            'dropout_rate': self.dropout_rate,
            'ops_order': self.ops_order,
        }
    @staticmethod
    def build_from_config(config):
        raise NotImplementedError
    def get_flops(self, x):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False

class ConvLayer(My2DLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 bias,
                 use_bn=True,
                 act_func='relu',
                 dropout_rate=0.,
                 ops_order='act_weight_bn'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = get_padding_size(self.kernel_size, 1)
        self.bias = bias
        self.groups = 1
        super(ConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

    def weight_op(self):
        weight_dict = OrderedDict()
        weight_dict['conv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride,
            padding=self.padding, dilation=1, groups=1, bias=self.bias
        )
        return weight_dict
    @property
    def module_str(self):
        return '{}x{}_Conv'.format(self.kernel_size, self.kernel_size)

    @property
    def config(self):
        return {
            'name': ConvLayer.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'bias': self.bias,
            **super(ConvLayer, self).config,
        }
    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)
    def get_flops(self, x):
        return count_conv_flop(self.conv, x), self.forward(x)

class DilConv(My2DLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dilation,
                 bias=False,
                 use_bn=True,
                 act_func='relu',
                 dropout_rate=0.,
                 ops_order='act_weight_bn'):
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = get_padding_size(self.kernel_size, self.dilation)
        self.bias = bias
        super(DilConv, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)
    def weight_op(self):
        weight_dict = OrderedDict()
        weight_dict['conv'] = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
            dilation=self.dilation, groups=1, bias=self.bias
        )
        return weight_dict
    @property
    def module_str(self):
        return '{}x{}d{}_DilatedConv'.format(self.kernel_size, self.kernel_size, self.dilation)
    @property
    def config(self):
        return {
            'name': DilConv.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'bias': self.bias,
            **super(DilConv, self).config
        }
    @staticmethod
    def build_from_config(config):
        return DilConv(**config)
    def get_flops(self, x):
        return count_conv_flop(self.conv, x), self.forward(x)

class SepConv(My2DLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 bias=False,
                 use_bn=True,
                 act_func='relu',
                 dropout_rate=0.,
                 ops_order='act_weight_bn'):

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = get_padding_size(self.kernel_size, 1)
        self.bias = bias
        super(SepConv, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)
    def weight_op(self):

        weight_dict = OrderedDict()
        weight_dict['dconv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
            dilation=1, groups=self.in_channels, bias=False,
        )
        weight_dict['pconv'] = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False
        )
        return weight_dict

    @property
    def module_str(self):
        return '{}x{}_SepConv'.format(self.kernel_size, self.kernel_size)
    @property
    def config(self):
        return {
            'name': SepConv.__name__,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'bias': self.bias,
            **super(SepConv, self).config
        }
    @staticmethod
    def build_from_config(config):
        return SepConv(**config)
    def get_flops(self, x):
        delta_flops1 = count_conv_flop(self.dconv, x)
        tmp = self.dconv(x)
        delta_flops2 = count_conv_flop(self.pconv, tmp)
        return (delta_flops1+delta_flops2), self.forward(x)


class Identity(My2DLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bn=False,
                 act_func=None,
                 dropout_rate=0.,
                 ops_order='act_weight_bn'):
        super(Identity, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)
        # pay attention to use_bn act_func of Identity Operation
    def weight_op(self):
        return None

    @property
    def module_str(self):
        return 'Identity'
    @property
    def config(self):
        return {
            'name': Identity.__name__,
            **super(Identity, self).config
        }

    @staticmethod
    def build_from_config(config):
        return Identity(**config)

    def get_flops(self, x):
        return 0, self.forward(x)


class Zero(MyModule):
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return torch.mul(x, 0.)
        return torch.mul(input[:, :, ::self.stride, ::self.stride], 0.)

    @property
    def module_str(self):
        return 'Zero'
    @property
    def config(self):
        return {
            'name': Zero.__name__,
            'stride': self.stride,
        }
    @staticmethod
    def build_from_config(config):
        Zero(**config)
    def get_flops(self, x):
        return 0, self.forward(x)

    @staticmethod
    def is_zero_layer():
        return True


class FactorizedReduce(MyModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 act='relu',
                 use_bn=True,
                 bias=False,
                 ops_order='act_weight_bn'):
        super(FactorizedReduce, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        assert out_channels % 2 == 0, 'FactorizedReduce Layer {} can not be divided by 2'.format(self.out_channels)
        self.bias=bias
        self.ops_order = ops_order
        self.act = act
        self.act_func = build_activation(act, inplace=False)
        # bn after weight_conv by default
        if use_bn:
            self.bn = nn.BatchNorm2d(out_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=1, stride=2, padding=0, bias=False)
    def forward(self, x):
        x = self.act_func(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

    @property
    def module_str(self):
        return 'FactorizedReduce'
    @property
    def config(self):
        return {
            'name': FactorizedReduce.__name__,
            'inc': self.in_channels,
            'outc': self.out_channels,
            'bias': self.bias,
            'act': self.act,
            'ops_order': self.ops_order
        }
    @staticmethod
    def build_from_config(config):
        return FactorizedReduce(**config)
    def get_flops(self, x):
        delta_flops1 = count_conv_flop(self.conv_1, x)
        delta_flops2 = count_conv_flop(self.conv_2, x[:, :, 1:, 1:])
        return (delta_flops1+delta_flops2), self.forward(x)
    @staticmethod
    def is_zero_layer():
        return False


class FactorizedIncrease(MyModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 act='relu',
                 use_bn=True,
                 bias=False,
                 ops_order='act_weight_bn'):
        super(FactorizedIncrease, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act
        self.use_bn = use_bn
        self.bias = bias
        self.ops_order = ops_order

        self.act_func = build_activation(self.act, inplace=False)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(self.out_channels)
        self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.upsample_layer(x)
        x = self.act_func(x)
        x = self.conv(x)
        x = self.bn(x)
        return x

    @property
    def module_str(self):
        return 'FactorizedIncrease'

    @property
    def config(self):
        return {
            'name': FactorizedIncrease.__name__,
            'inc': self.in_channels,
            'out': self.out_channels,
            'act_func': self.act,
            'use_bn': self.use_bn,
            'bias': self.bias,
            'ops_order': self.ops_order
        }
    @staticmethod
    def build_from_config(config):
        FactorizedIncrease(**config)

    def get_flops(self, x):
        tmp_x = self.upsample_layer(x)
        delta_flops = count_conv_flop(self.conv, tmp_x)
        return delta_flops, self.forward(x)

    @staticmethod
    def is_zero_layer():
        return False

class ASPP(MyModule):
    def __init__(self, in_channels, out_channels, dilation, nb_classes):
        super(ASPP, self).__init__()

        self.nb_classes = nb_classes
        self.dilation = dilation
        self.conv1x1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, in_channels, 1, bias=False)),
            ('bn', nn.BatchNorm2d(in_channels))
        ]))
        padding3x3 = get_padding_size(3, dilation)
        self.conv3x3 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, in_channels, 3, padding=padding3x3, dilation=dilation, bias=False)),
            ('bn', nn.BatchNorm2d(in_channels)),
        ]))
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.gp_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels, in_channels, 1, bias=False)),
            ('bn', nn.BatchNorm2d(in_channels)),
            ('relu', nn.ReLU(inplace=False))
        ]))
        self.concat_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(in_channels * 3, in_channels, 1, bias=False)),
            ('bn', nn.BatchNorm2d(in_channels))
        ]))
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, input):

        conv1x1 = self.conv1x1(input)
        conv3x3 = self.conv3x3(input)
        convgp = self.global_pooling(input)
        convgp = F.interpolate(convgp, size=input.size()[2:], mode='bilinear', align_corners=True)
        convgp = self.gp_conv(convgp)

        concat = torch.cat([conv1x1, conv3x3, convgp], dim=1)
        return self.final_conv(self.concat_conv(concat))

    def get_flops(self, x):

        flop_conv1x1, conv1x1 = count_normal_conv_flop(self.conv1x1.conv,x), self.conv1x1(x)
        flop_conv3x3, conv3x3 = count_normal_conv_flop(self.conv3x3.conv, x), self.conv3x3(x)
        convgp = F.interpolate(self.global_pooling(x), size=x.size()[2:], mode='bilinear', align_corners=True)
        flop_convgp, convgp = count_normal_conv_flop(self.gp_conv.conv, convgp), self.gp_conv(convgp)
        cat_feature = torch.cat([conv1x1, conv3x3, convgp], dim=1)
        flop_concat, output = count_normal_conv_flop(self.concat_conv.conv, cat_feature), self.concat_conv(cat_feature)
        flop_final, output = count_normal_conv_flop(self.final_conv, output), self.final_conv(output)
        return flop_conv1x1 + flop_conv3x3 + flop_convgp + flop_concat + flop_final, output

    def module_str(self):
        return 'ASPP conv1x1 conv3x3_d{} globpooling'.format(3, 3, self.dilation)


class MBInvertedConvLayer(MyModule):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim)),
                ('act', nn.ReLU6(inplace=False))
            ]))

        pad = get_padding_size(self.kernel_size, 1)

        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim)),
            ('act', nn.ReLU6(inplace=False))
        ]))

        self.point_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))
    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

    # TODO: remove @property
    @property
    def module_str(self):
        return '{}x{}_MBConv{}'.format(self.kernel_size, self.kernel_size, self.expand_ratio)

    @property
    def config(self):
        return {
            'name': MBInvertedConvLayer.__name__,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'expand_ratio': self.expand_ratio,
            'mid_channels': self.mid_channels
        }
    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)

    def get_flops(self, x):
        if self.inverted_bottleneck:
            flop1 = count_conv_flop(self.inverted_bottleneck.conv, x)
            x = self.inverted_bottleneck(x)
        else:
            flop1 = 0

        flop2 = count_conv_flop(self.depth_conv.conv, x)
        x = self.depth_conv(x)
        flop3 = count_conv_flop(self.point_conv.conv, x)
        x = self.point_conv(x)

        return flop1 + flop2 + flop3, x

    @staticmethod
    def is_zero_layer():
        return False

class MobileInvertedResidualBlock(MyModule):
    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    def forward_gdas(self, x, weight, index):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv.forward_gdas(x, weight, index)
        else:
            conv_x = self.mobile_inverted_conv.forward_gdas(x, weight, index)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    def module_str(self):
        # self.mobile_inverted_conv is MixedOperation
        # shortcut is Identity
        return 'MBConvResBlock({}, {})'.format(
            self.mobile_inverted_conv.module_str() if self.mobile_inverted_conv is not None else None,
            self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)
