import torch.nn.functional as F

from modules.operations import *
from modules.mixed_op import *
from genotype import PRIMITIVES

from run_manager import *

class Cell(MyModule):
    def __init__(self,
                 run_config: RunConfig,
                 conv_candidates,
                 scale, prev_c, prev_prev_c, types):
        super(Cell, self).__init__()
        self.run_config = run_config
        self.conv_candidates = conv_candidates
        self.scale = scale
        self.prev_c = prev_c
        self.prev_prev_c = prev_prev_c
        self.types = types

        self.filter_multiplier = self.run_config.filter_multiplier
        self.block_multiplier = self.run_config.block_multiplier
        self.steps = self.run_config.steps
        self.ops = nn.ModuleList()

        # according to paper
        self.outc = int(self.block_multiplier * self.filter_multiplier * self.scale / 4)

        if prev_prev_c is not None:
            if prev_prev_c == -1: # by computing
                self.prev_prev_c = int(self.block_multiplier * self.filter_multiplier * self.scale / 4)
                self.preprocess0 = ConvLayer(self.prev_prev_c, self.outc,
                                             kernel_size=1, stride=1, bias=0)
            else: # fixed
                self.prev_prev_c = prev_prev_c
                self.preprocess0 = ConvLayer(self.prev_prev_c, self.outc,
                                             kernel_size=1, stride=1, bias=False)

        if prev_c is not None:
            if prev_c == -1: # by computing
                for type in self.types:
                    if 'reduction' in type:
                        self.prev_c_down = int(self.block_multiplier * self.filter_multiplier * self.scale /2 /4)
                        self.preprocess1_down = FactorizedReduce(self.prev_c_down, self.outc)
                    if 'same' in type:
                        self.prev_c_same = int(self.block_multiplier * self.filter_multiplier * self.scale / 4)
                        self.preprocess1_same = ConvLayer(self.prev_c_same, self.outc,
                                                          kernel_size=1, stride=1, bias=False)
                    if 'up' in type:
                        self.prev_c_up = int(self.block_multiplier * self.filter_multiplier * self.scale * 2 / 4)
                        self.preprocess1_up = FactorizedIncrease(self.prev_c_up, self.outc)
            else: # fixed
                self.prev_c_down = prev_c
                self.prev_c_same = prev_c
                self.prev_c_up = prev_c
                for type in self.types:
                    if 'reduction' in type:
                        self.preprocess1_down = FactorizedReduce(self.prev_c_down, self.outc)
                    if 'same' in type:
                        self.preprocess1_same = ConvLayer(self.prev_c_same, self.outc, kernel_size=1, stride=1, bias=False)
                    if 'up' in type:
                        self.preprocess1_up = FactorizedIncrease(self.prev_c_up, self.outc)
        for i in range(self.steps):
            for j in range(i+2):
                stride = 1
                if self.prev_prev_c is None and j == 0: # the first mixededge related to prev_prev_cell
                    op = None
                else:
                    # skip connection: Identity
                    # None: Zero
                    op = MixedEdge(build_candidate_ops(
                        self.conv_candidates,
                        in_channels=self.outc, out_channels=self.outc,
                        stride=stride, ops_order='act_weight_bn'
                    ))
                self.ops.append(op)
        self.final_conv1x1 = ConvLayer(self.steps * self.outc, self.outc, 1, 1, 0)

    def forward(self, s0, s1_down, s1_same, s1_up,):# weights):
        # TODO: get rid of useless weights
        all_states = []
        if s0 is not None:
            s0 = self.preprocess0(s0)

        if s1_down is not None:
            s1_down = self.preprocess1_down(s1_down)
            state_down = [s0, s1_down]
            all_states.append(state_down)
        if s1_same is not None:
            s1_same = self.preprocess1_same(s1_same)
            state_same = [s0, s1_same]
            all_states.append(state_same)
        if s1_up is not None:
            s1_up = self.preprocess1_up(s1_up)
            state_up = [s0, s1_up]
            all_states.append(state_up)

        final_concates = []
        for states in all_states:
            # [s0, s1_down] [s1, s1_same] [s0, s1_up] in order
            offset = 0
            # within a specific cell
            for i in range(self.steps):
                new_states = [] # output of each path of a node
                # for a specific node
                for idx, h in enumerate(states): # idx means branch_index within a cell, h means hidden_states
                    branch_index = offset + idx
                    if h is None or self.ops[branch_index] is None: # hidden_state is None or operation is None
                        continue
                    # TODO: here has change, mixed_ops in darts and proxylessNAS
                    # the latter one does not have operation importance
                    new_state = self.ops[branch_index](h) # output of each path of a node, only active one operation
                    new_states.append(new_state) # append outputs of all the paths of a node
                s = sum(new_states) # node output
                offset += len(states) # operations are appended in order
                states.append(s) # as previous output for the following nodes

            # states: prev_prev_c_output, prevc_output, nodes output
            concat_feature = torch.cat(states[-self.steps:], dim=1)
            concat_feature = self.final_conv1x1(concat_feature)
            final_concates.append(concat_feature)

        # final_concates have three results at most, for three paths, respectively.
        return final_concates


