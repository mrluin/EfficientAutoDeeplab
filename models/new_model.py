import torch
import torch.nn as nn
import numpy as np


from modules.operations import *
from modules.my_modules import *
from utils.common import *

# get cell_arch and network_arch

class SplitCell(MyModule):
    def __init__(self,
                 layer, filter_multiplier, block_multiplier, steps,
                 next_scale, prev_prev_scale, prev_scale,
                 cell_arch, network_arch,
                 args):
        super(SplitCell, self).__init__()

        self.cell_arch = cell_arch
        self.network_arch = network_arch
        self.args = args
        # prev_prev_scale -> prev_scale -> next_scale
        self.out_channels = int(filter_multiplier * block_multiplier * prev_scale / 4)
        if prev_prev_scale is not None:
            if layer == 0:
                prev_prev_c = 64
            elif layer == 1:
                prev_prev_c = 128
            else:
                prev_prev_c = int(filter_multiplier * block_multiplier * prev_prev_scale / 4)
            if prev_prev_scale == next_scale:
                self.preprocess0 = ConvLayer(in_channels=prev_prev_c, out_channels=self.out_channels,
                                             kernel_size=1, stride=1, bias=False)
            elif prev_prev_scale == next_scale - 1:
                self.preprocess0 = FactorizedReduce(in_channels=prev_prev_c, out_channels=self.out_channels)
            elif prev_prev_scale == next_scale + 1:
                self.preprocess0 = FactorizedIncrease(in_channels=prev_prev_c, out_channels=self.out_channels)
            else:
                raise ValueError('invalid relation between prev_prev_scale and next_scale')
        if prev_scale is not None:
            if layer == 0:
                prev_c = 128
            else:
                prev_c = int(filter_multiplier * block_multiplier * prev_scale / 4)
            if prev_scale == next_scale - 1:
                self.preprocess1 = FactorizedReduce(prev_c, self.out_channels)
            elif prev_scale == next_scale:
                self.preprocess1 = ConvLayer(prev_c, self.out_channels, 1, 1, False)
            elif prev_scale == next_scale + 1:
                self.preprocess1 = FactorizedIncrease(prev_c, self.out_channels)

        self.steps = steps
        self.ops = nn.ModuleList()

        for x in self.cell_arch:

            # TODO: cell architecture decode, and build cell according to derived PRIMITIVES

    def forward(self, prev_prev_c ,prev_c):
        s0 = prev_prev_c
        s1 = prev_c

        # TODO: cell forward


class NewNetwork(MyNetwork):
    def __init__(self, network_arch, cell_arch,
                 filter_multiplier, block_multiplier, steps,
                 nb_layers, nb_classes, init_channels,
                 cell=SplitCell,):
        super(NewNetwork, self).__init__()

        self.cell_arch = torch.from_numpy(cell_arch)
        self.network_arch = torch.from_numpy(network_arch)

        self.filter_multiplier = filter_multiplier
        self.block_multiplier = block_multiplier
        self.steps = steps

        self.nb_classes = nb_classes
        self.nb_layers = nb_layers
        self.init_channels = init_channels # from layer0 scale0 128
        half_init_channels = self.init_channels // 2

        self.stem0 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, half_init_channels, 3, 2, 1)),
            ('bn', nn.BatchNorm2d(half_init_channels))
        ]))
        self.stem1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(half_init_channels, half_init_channels, 3, stride=1, padding=1)),
            ('bn', nn.BatchNorm2d(half_init_channels))
        ]))
        self.stem2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(half_init_channels, init_channels, 3, 2, 1)),
            ('bn', nn.BatchNorm2d(init_channels))
        ]))
        self.cells = nn.ModuleList()
        # network_arch 12, 4, 3, from layer to scale choose which operation
        # prev_scale is current layer scale, w.r.t. next's prev
        # prev_prev_scale is previous layer scale, w.r.t. next's prev_prev

        prev_prev_scale = 0
        prev_scale = 0
        for i in range(self.nb_layers):
            if i == 0:
                next_scale_option = torch.sum(self.network_arch[i], dim=1)
                next_scale = torch.argmax(next_scale_option).item()
                prev_scale = 0
                prev_prev_scale = -1
            elif i == 1:
                next_scale_option = torch.sum(self.network_arch[i], dim=1)
                prev_scale_option = torch.sum(self.network_arch[i - 1], dim=1)
                next_scale = torch.argmax(next_scale_option).item()
                prev_scale = torch.argmax(prev_scale_option).item()
                prev_prev_scale = 0
            else:
                next_scale_option = torch.sum(self.network_arch[i], dim=1)
                prev_scale_option = torch.sum(self.network_arch[i-1], dim=1)
                prev_prev_scale_option = torch.sum(self.network_arch[i-2], dim=1)
                next_scale = torch.argmax(next_scale_option).item()
                prev_scale = torch.argmax(prev_scale_option).item()
                prev_prev_scale = torch.argmax(prev_prev_scale_option).item()
            assert next_scale < (i + 1), 'invalid next_scale in layer {}'.format(i)

            _cell = cell(i, self.filter_multiplier, self.block_multiplier, self.steps, next_scale,
                         prev_prev_scale=prev_prev_scale, prev_scale=prev_scale,
                         cell_arch=self.cell_arch, network_arch=self.network_arch, args=self.args)

            self.cells += [_cell]

    def forward(self, x):
        stem = self.stem0(x)
        stem0 = self.stem1(stem)
        stem1 = self.stem2(stem0)

        inter_features = [stem0, stem1]
        for i in range(self.nb_layers):




        # TODO: construct cell and network route, according to network_arch and cell_arch


        '''
        # network_arch and cell_arch
        
        network_arch, cell_arch, and network_path
        
        
        network_arch :: network_layer_to_space()
        cell_arch
        network_path :: len()=12
        '''


def network_layer_to_space(net_arch):
    # net_arch = [1, 0, 0, 1, 2, 1, 2, 2, 3, 3, 2, 1]
    # space[layer][scale][choice] means from layer to next_scale choice sample
    prev = 0
    space = None
    for i, scale in enumerate(net_arch):
        if i == 0:
            space = np.zeros((1, 4, 3))
            if scale == prev + 1:
                sample = 2
            elif scale == prev:
                sample = 1
            elif scale == prev - 1:
                sample = 0
            else:
                raise NotImplementedError
            space[0][scale][sample] = 1
            prev = scale
        else:
            # modification on relation of sample and scales
            if scale == prev + 1: # down
                sample = 2
            elif scale == prev: # same
                sample = 1
            elif scale == prev - 1: # up
                sample = 0
            else:
                raise NotImplementedError
            space1 = np.zeros((1, 4, 3))
            space1[0][scale][sample] = 1
            space = np.concatenate([space, space1], axis=0)
            prev = scale

    return space

def genotype_decode(net):
    # according to probs_over_paths, obtaining the largest value path
    # for proxylessNAS, each mixed edge, only one operation

    # TODO: specific architecture for each cell

    # 1. how many cells     can be calculated, w.r.t. layers
    # 2. how many steps     known
    # 3. how many edges --> how many mixed operations, can be calculated by
    # 4. in darts, if steps = 5 in each cell. Finally, it will get 2 * 5 paths, i.e. each node get the max two prob path

        # TODO: each path have only one operation, and then each node get only one path
    # when nb_layer >= 2
    nb_cells = 5 + (nb_layer - 2) * 4
    nb_edges = torch.sum([_i for _i in range(steps) for _j in range(_i+2)])

    '''
    # under the case of, the super_network is already exist
    '''
    for i in net._modules:
        if i == 'cells':
            child = net._modules[i]
            for j in child._modules:
                child_child = child._modules[j]  # each cell
                for k in child_child._modules:
                    if k == 'ops':
                        child_child_child = child_child._modules[k]  # ops in each cell
                        for m in child_child_child._modules:
                            child_child_child_child = child_child_child._modules[m]  # MBResidualBlocks in each ops in each cell
                            if child_child_child_child is not None:
                                for n in child_child_child_child._modules:
                                    cc = child_child_child_child._modules[n]
                                    if cc.__str__().startswith('MixedEdge'):
                                        child_child_child_child._modules[n] = cc.chosen_op

    '''
    # under the case of get cell gene[]
    '''
    def _parse(alphas, steps):
        gene = []
        start = 0
        n = 2 # offset
        for i in range(steps):
            end = start + n
            # all the edge ignore None operation
            edges = sorted(range(start, end), key=lambda x:-np.max(alphas[x, 1:]))
            top1edge = edges[0] # edge index
            best_op_index = np.argmax(alphas[top1edge]) #
            gene.append([top1edge, best_op_index])
            start = end
            n += 1 # move offset
        return np.array(gene)

    # TODO: apply _parse to each cell in super_network
    '''
    # each cell in the super_network has an array of gene, save len[]=nb_cellls
    # when construct super_network, get index according to layer and scale information,
    # when construct cell_architecture, according to super_gene[index]
    '''