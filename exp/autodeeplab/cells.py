# author: Jingbo Lin
# contact: ljbxd180612@gmail.com
# github: github.com/mrluin

import torch
import torch.nn as nn
import torch.nn.functional as F


from modules.my_modules import MyModule
from modules.operations import ConvLayer, FactorizedReduce, DoubleFactorizedReduce, FactorizedIncrease, Identity, MobileInvertedResidualBlock
from models.gumbel_cells import MixedOp, build_candidate_ops, proxyless, autodeeplab, my_search_space

class NormalCell(MyModule):
    def __init__(self, layer, filter_multiplier, block_multiplier, steps, scale, search_space,
                 pps=None, ps=None, affine=True):
        super(NormalCell, self).__init__()
        self.layer = layer
        self.filter_multiplier = filter_multiplier
        self.block_multiplier = block_multiplier
        self.steps = steps
        self.scale = scale
        self.search_space = search_space
        if search_space == 'proxyless':
            self.conv_candidates = proxyless
        elif search_space == 'autodeeplab':
            self.conv_candidates = autodeeplab
        elif search_space == 'my_search_space':
            self.conv_candidates = my_search_space

        self.affine = affine

        self.index2channel = {
            0: 32,
            1: 64,
            2: 128,
            3: 256,
        }
        self.total_nodes = 2 + self.steps

        self.outc = self.index2channel[self.scale]

        # pps: prev_prev_scale
        # ps : prev_scale
        if self.index2channel.get(pps) is not None:
            self.ppc = self.index2channel[pps]
            if pps == scale:
                self.preprocess0 = ConvLayer(self.ppc, self.outc, 1, 1, False, affine=self.affine)
            elif scale - pps == 1:
                self.preprocess0 = FactorizedReduce(self.ppc, self.outc, affine=self.affine)
            elif scale - pps == 2:
                self.preprocess0 = DoubleFactorizedReduce(self.ppc, self.outc, affine=self.affine)
            else:
                raise ValueError('invalid relationship in ppc, layer.scale {:}.{:}'.format(layer, scale))
        else:
            self.ppc = pps
            if layer == 0 and scale == 0:
                self.preprocess0 = FactorizedReduce(self.ppc, self.outc, affine=self.affine)
            elif layer == 1 and scale == 0:
                self.preprocess0 = ConvLayer(self.ppc, self.outc, 1, 1, False, affine=self.affine)
            elif layer == 0 and scale == 1:
                self.preprocess0 = DoubleFactorizedReduce(self.ppc, self.outc, affine=self.affine)
            elif layer == 1 and scale == 1:
                self.preprocess0 = FactorizedReduce(self.ppc, self.outc, affine=self.affine)
            elif layer == 1 and scale == 2:
                self.preprocess0 = DoubleFactorizedReduce(self.ppc, self.outc, affine=self.affine)
            else:
                raise ValueError('invalid relationship in ppc, layer.scale {:}.{:}'.format(layer, scale))

        if self.index2channel.get(ps) is not None:
            self.pc = self.index2channel[ps]
            if scale == ps:
                self.preprocess1 = ConvLayer(self.pc, self.outc, 1, 1, False, affine=self.affine)
            elif scale - ps == 1:
                self.preprocess1 = FactorizedReduce(self.pc, self.outc, affine=self.affine)
            elif scale - ps == -1:
                self.preprocess1 = FactorizedIncrease(self.pc, self.outc, affine=self.affine)
            else:
                raise ValueError('invalid relationship in pc, layer.scale {:}.{:}'.format(layer, scale))
        else:
            self.pc = ps
            if layer == 0 and scale == 0:
                self.preprocess1 = ConvLayer(self.pc, self.outc, 1, 1, False, affine=self.affine)
            elif layer == 0 and scale == 1:
                self.preprocess1 = FactorizedReduce(self.pc, self.outc, affine=self.affine)
            else:
                raise ValueError('invalid relationship in pc, layer.scale {:}.{:}'.format(layer, scale))

        self.ops = nn.ModuleDict()
        # i::node_index, j::previous_node_index
        if self.search_space == 'proxyless':
            for i in range(2, self.total_nodes):
                for j in range(i):
                    edge_str = '{:}<-{:}'.format(i, j)
                    mobile_inverted_conv = MixedOp(
                        build_candidate_ops(self.conv_candidates,
                                            in_channels=self.outc, out_channels=self.outc, stride=1,
                                            ops_order='act_weight_bn', affine=self.affine))  # normal MixedOp, ModuleList with weight
                    shortcut = Identity(self.outc, self.outc)
                    inverted_residual_block = MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)
                    self.ops[edge_str] = inverted_residual_block
        elif self.search_space == 'autodeeplab' or self.search_space == 'my_search_space':
            for i in range(2, self.total_nodes):
                for j in range(i):
                    edge_str = '{:}<-{:}'.format(i, j)

                    op = MixedOp(build_candidate_ops(self.conv_candidates, in_channels=self.outc, out_channels=self.outc, stride=1,
                                                ops_order='act_weight_bn', affine=self.affine))
                    self.ops[edge_str] = op
        else:
            raise ValueError('search space {:} is not supported'.format(self.search_space))

        self.finalconv1x1 = ConvLayer(self.steps * self.outc, self.outc, 1, 1, False)
        self.edge_keys = sorted(list(self.ops.keys())) # 'sorted by {:}<-{:}'
        self.edge2index = {key:i for i, key in enumerate(self.edge_keys)} # {:}<-{:} : index
        self.nb_edges = len(self.ops)

        # for un-shared cell structure
        #self.cell_arch_parameters = nn.Parameter(1e-3 * torch.randn(self.nb_edges, self.n_choice))

    @property
    def n_choice(self):
        return len(self.conv_candidates)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)
        states = [s0, s1]
        for i in range(2, self.total_nodes):
            new_states = []
            for j in range(i):
                edge_str = '{:}<-{:}'.format(i, j)
                branch_index = self.edge2index[edge_str]
                related_hidden = states[j]
                weight = weights[branch_index]
                new_state = self.ops[edge_str](related_hidden, weight)
                new_states.append(new_state)
            s = sum(new_states)
            states.append(s)

        concat_feature = torch.cat(states[-self.steps:], dim=1)
        concat_feature = self.finalconv1x1(concat_feature)
        return concat_feature
