# ===============================
# author : Jingbo Lin
# contact: ljbxd180612@gmail.com
# github : github.com/mrluin
# ===============================

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from collections import OrderedDict

class nonbt_dw(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 ):
        super(nonbt_dw, self).__init__()

        self.conv1_dw = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=(stride, stride),
                                 padding=(int((kernel_size-1)/2),int((kernel_size-1)/2)),
                                 dilation =(1,1),
                                 groups=in_channels, bias=False)
        self.conv1_dw_bn = nn.BatchNorm2d(out_channels)
        self.conv1_pw = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.conv1_pw_bn = nn.BatchNorm2d(out_channels)
        # here is relu
        self.conv2_dw = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=(stride, stride),
                                 padding=(int((kernel_size-1)/2*dilation),int((kernel_size-1)/2*dilation)),
                                 dilation =(dilation, dilation),
                                 groups=in_channels, bias=False)
        self.conv2_dw_bn = nn.BatchNorm2d(out_channels)
        self.conv2_pw = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.conv2_pw_bn = nn.BatchNorm2d(out_channels)

        # here is relu
    def forward(self, x):
        residual = x
        m = self.conv1_dw(x)
        m = self.conv1_dw_bn(m)
        m = self.conv1_pw(m)
        m = self.conv1_pw_bn(m)

        m = self.conv2_dw(F.relu(m, inplace=True))
        m = self.conv2_dw_bn(m)
        m = self.conv2_pw(m)
        m = self.conv2_pw_bn(m)

        return F.relu(torch.add(m, residual), inplace=True)


class Separabel_conv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 kernel_size=(3,3),
                 dilation=(1,1),
                 #padding=(1,1),
                 stride=(1,1),
                 bias=False):
        """
        # Note: Default for kernel_size=3,
        for Depthwise conv2d groups should equal to in_channels and out_channels == in_channels
        Only bn after depthwise_conv2d and no no-linear
        padding = (kernel_size-1) / 2
        padding = padding * dilation
        """
        super(Separabel_conv2d, self).__init__()
        self.depthwise_conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding= (int((kernel_size[0]-1)/2)*dilation[0],int((kernel_size[1]-1)/2)*dilation[1]),
            dilation= dilation, groups=groups,bias=bias
        )
        self.dw_bn = nn.BatchNorm2d(out_channels)
        self.pointwise_conv2d = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1, padding=0, dilation=1, groups=1, bias=False
        )
        self.pw_bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):

        out = self.depthwise_conv2d(input)
        out = self.dw_bn(out)
        out = self.pointwise_conv2d(out)
        out = self.pw_bn(out)

        return out

class down_sampling_block(nn.Module):

    def __init__(self, inpc, oupc):
        super(down_sampling_block, self).__init__()
        self.branch_conv = nn.Conv2d(inpc, oupc-inpc, 3, stride=2, padding= 1, bias=False)
        self.branch_mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(oupc, eps=1e-03)

    def forward(self, x):
        output = torch.cat([self.branch_conv(x), self.branch_mp(x)], 1)
        output = self.bn(output)

        return F.relu(output)

class bt_dw(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 ):

        # default decoupled
        super(bt_dw, self).__init__()

        self.internal_channels = in_channels // 4
        # compress conv
        self.conv1 = nn.Conv2d(in_channels, self.internal_channels, 1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(self.internal_channels)
        # a relu
        self.conv2_3 = nn.Conv2d(self.internal_channels, self.internal_channels, (kernel_size, kernel_size), stride=(stride, stride),
                                 padding=(int((kernel_size-1)/2*dilation),int((kernel_size-1)/2*dilation)),
                                 dilation =(dilation, dilation),
                                 groups=self.internal_channels, bias=False)
        self.bn_2_3 = nn.BatchNorm2d(self.internal_channels)
        # a relu
        self.conv4 = nn.Conv2d(self.internal_channels, out_channels, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):

        residual = input
        main = self.conv1(input)
        main = self.conv2_3(F.relu(self.conv1_bn(main), inplace=True))
        main = self.conv4(F.relu(self.bn_2_3(main), inplace=True))

        return F.relu(torch.add(self.conv4_bn(main),residual))

class nonbt_dw_fac(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 dropout_rate=0.0,
                 ):
        super(nonbt_dw_fac, self).__init__()

        # defaultly inchannels = outchannels
        self.conv1_1 = nn.Conv2d(in_channels, out_channels, (kernel_size,1), stride=(stride,1),
                                padding=(int((kernel_size-1)/2),0), dilation=(1,1),
                                groups=in_channels, bias=False)
        self.conv1_1_bn = nn.BatchNorm2d(out_channels)
        self.conv1_2 = nn.Conv2d(in_channels, out_channels, (1,kernel_size), stride=(1,stride),
                                 padding=(0,int((kernel_size-1)/2)), dilation=(1, 1),
                                 groups=in_channels, bias=False)
        self.conv1_2_bn = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.conv1_bn = nn.BatchNorm2d(out_channels)
        # here is relu

        self.conv2_1 = nn.Conv2d(in_channels, out_channels, (kernel_size,1), stride=(stride,1),
                                padding=(int((kernel_size-1)/2*dilation),0), dilation=(dilation,1),
                                groups=in_channels, bias=False)
        self.conv2_1_bn = nn.BatchNorm2d(out_channels)
        self.conv2_2 = nn.Conv2d(in_channels, out_channels, (1,kernel_size), stride=(1,stride),
                                 padding=(0,int((kernel_size-1)/2*dilation)), dilation=(1, dilation),
                                 groups=in_channels, bias=False)
        self.conv2_2_bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False)
        self.conv2_bn = nn.BatchNorm2d(out_channels)
        #self.drop2d = nn.Dropout2d(p=dropout_rate)

        # here is relu
    def forward(self, x):

        residual = x
        main = self.conv1_1(x)
        main = self.conv1_1_bn(main)
        main = self.conv1_2(main)
        main = self.conv1_2_bn(main)
        main = self.conv1(main)
        main = self.conv1_bn(main)

        main = self.conv2_1(main)
        main = self.conv2_1_bn(main)
        main = self.conv2_2(main)
        main = self.conv2_2_bn(main)
        main = self.conv2(main)

        return F.relu(torch.add(self.conv2_bn(main), residual), inplace=True)

class SFRB(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 dropout_rate =0.0,
                 ):

        # default decoupled
        super(SFRB, self).__init__()

        self.internal_channels = in_channels // 4
        # compress conv
        self.conv1 = nn.Conv2d(in_channels, self.internal_channels, 1, bias=False)
        self.conv1_bn = nn.BatchNorm2d(self.internal_channels)
        # a relu
        # Depthwise_conv 3x1 and 1x3
        self.conv2 = nn.Conv2d(self.internal_channels, self.internal_channels, (kernel_size,1), stride=(stride,1),
                               padding=(int((kernel_size-1)/2*dilation),0), dilation=(dilation,1),
                               groups=self.internal_channels, bias=False)
        self.conv2_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv3 = nn.Conv2d(self.internal_channels, self.internal_channels, (1,kernel_size), stride=(1,stride),
                               padding=(0,int((kernel_size-1)/2*dilation)), dilation=(1, dilation),
                               groups=self.internal_channels, bias=False)
        self.conv3_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv4 = nn.Conv2d(self.internal_channels, out_channels, 1, bias=False)
        self.conv4_bn = nn.BatchNorm2d(out_channels)

        # regularization
        self.dropout = nn.Dropout2d(inplace=True, p=dropout_rate)

    def forward(self, input):

        residual = input
        main = self.conv1(input)
        main = self.conv1_bn(main)
        main = F.relu(main, inplace=True)

        main = self.conv2(main)
        main = self.conv2_bn(main)
        main = self.conv3(main)
        main = self.conv3_bn(main)
        main = self.conv4(main)
        main = self.conv4_bn(main)

        if self.dropout.p != 0:
            main = self.dropout(main)

        return F.relu(torch.add(main, residual), inplace=True)

class DownSamplingBlock_v2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,):
        """
        # Note: Initial_block from ENet
        Default that out_channels = 2 * in_channels
        compared to downsamplingblock_v1, change conv3x3 into Depthwise and projection_layer
        Add: channel_shuffle after concatenate
        to be testing
        gc prelu/ relu
        """
        super(DownSamplingBlock_v2, self).__init__()

        # MaxPooling or AvgPooling
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        '''
        # FacDW
        self.depthwise_conv2d_1 = nn.Conv2d(in_channels= in_channels, out_channels=in_channels, kernel_size=(3,1), stride=(2,1),
                                            padding=(1,0), groups=in_channels, bias=False)
        self.dwbn1 = nn.BatchNorm2d(in_channels)
        self.depthwise_conv2d_2 = nn.Conv2d(in_channels= in_channels, out_channels=in_channels, kernel_size=(1,3), stride=(1,2),
                                            padding=(0,1), groups=in_channels, bias=False)
        self.dwbn2 = nn.BatchNorm2d(in_channels)
        '''
        self.depthwise_conv2d = nn.Conv2d(in_channels= in_channels, out_channels=in_channels,
                                          kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.dw_bn = nn.BatchNorm2d(in_channels)

        self.project_layer = nn.Conv2d(in_channels= in_channels, out_channels=out_channels-in_channels,
                                       kernel_size=1, bias=False)
        # here dont need project_bn, need to bn with ext_branch
        #self.project_bn = nn.BatchNorm2d(out_channels-in_channels)

        self.ret_bn = nn.BatchNorm2d(out_channels)
        self.ret_prelu = nn.ReLU(inplace= True)

    def forward(self, input):

        ext_branch = self.pooling(input)
        '''
        # facDW
        main_branch = self.dwbn1(self.depthwise_conv2d_1(input))
        main_branch = self.dwbn2(self.depthwise_conv2d_2(main_branch))
        '''
        main_branch = self.depthwise_conv2d(input)
        main_branch = self.dw_bn(main_branch)

        main_branch = self.project_layer(main_branch)

        ret = torch.cat([ext_branch, main_branch], dim=1)
        ret = self.ret_bn(ret)

        #ret = channel_shuffle(ret, 2)

        return self.ret_prelu(ret)

class ConvBnReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, bias=False):
        super(ConvBnReLu, self).__init__()

        self.cbr = nn.Sequential(OrderedDict({
                    'conv': nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
                    'bn': nn.BatchNorm2d(out_channels),
                    'relu': nn.ReLU(),
                }))
        self._weight_init()

    def forward(self, x):

        return self.cbr(x)

    def _weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, inplace=False):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.cbr = ConvBnReLu(in_channels, inter_channels)
        self.conv1x1 = nn.Conv2d(inter_channels, out_channels, 1, 1, 0, bias=False)

        self._weight_init()
    def forward(self, x):
        x = self.cbr(x)
        x = self.conv1x1(x)
        return x

    def _weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                init.constant_(module.weight, 1)
                init.constant_(module.bias, 0)

