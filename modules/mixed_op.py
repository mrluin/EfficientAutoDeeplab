import numpy as np
import torch.nn.functional as F

from modules.my_modules import MyModule
from torch.nn.parameter import Parameter
from modules.operations import *
from utils.common import delta_ij, detach_variable

def build_candidate_ops(candiate_ops, in_channels, out_channels, stride, ops_order):

    if candiate_ops is None:
        raise ValueError('Please specify a candidate set')

    # None zero layer
    name2ops = {
        'Identity': lambda inc, outc, s: Identity(inc, outc, ops_order=ops_order),
        'Zero': lambda inc, outc, s: Zero(s),
    }
    # add MBConv Layers
    name2ops.update({
        '3x3_MBConv1': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 3, s, 1),
        '3x3_MBConv2': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 3, s, 2),
        '3x3_MBConv3': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 3, s, 3),
        '3x3_MBConv4': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 3, s, 4),
        '3x3_MBConv5': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 3, s, 5),
        '3x3_MBConv6': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 3, s, 6),
        '5x5_MBConv1': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 5, s, 1),
        '5x5_MBConv2': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 5, s, 2),
        '5x5_MBConv3': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 5, s, 3),
        '5x5_MBConv4': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 5, s, 4),
        '5x5_MBConv5': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 5, s, 5),
        '5x5_MBConv6': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 5, s, 6),
        '7x7_MBConv1': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 7, s, 1),
        '7x7_MBConv2': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 7, s, 2),
        '7x7_MBConv3': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 7, s, 3),
        '7x7_MBConv4': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 7, s, 4),
        '7x7_MBConv5': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 7, s, 5),
        '7x7_MBConv6': lambda inc, outc, s: MBInvertedConvLayer(inc, outc, 7, s, 6),
        #===========================================================================
        '3x3_DWConv': lambda inc, outc, s: SepConv(inc, outc, 3, s),
        '5x5_DWConv': lambda inc, outc, s: SepConv(inc, outc, 5, s),
        '3x3_DilConv': lambda inc, outc, s: DilConv(inc, outc, 3, s, 2),
        '5x5_DilConv': lambda inc, outc, s: DilConv(inc, outc, 5, s, 2),
        '3x3_AvgPooling': lambda inc, outc, s: nn.AvgPool2d(3, stride=s, padding=1, count_include_pad=False),
        '3x3_MaxPooling': lambda inc, outc, s: nn.MaxPool2d(3, stride=s, padding=1),
    })
    return [
        name2ops[name](in_channels, out_channels, stride) for name in candiate_ops
    ]

class MixedEdge(MyModule):
    MODE = None
    def __init__(self, candidate_ops):
        super(MixedEdge, self).__init__()

        self.candidate_ops = nn.ModuleList(candidate_ops)
        # architecture parameter and binary gates
        self.AP_path_alpha = Parameter(torch.Tensor(self.n_choices))
        self.AP_path_wb = Parameter(torch.Tensor(self.n_choices))

        self.active_index = [0]
        self.inactive_index = None

        self.log_prob = None
        self.current_prob_over_ops = None

    @property
    def n_choices(self):
        # return total number of candidate operations
        return len(self.candidate_ops)

    @property
    def probs_over_ops(self):
        # return softmax probability
        probs = F.softmax(self.AP_path_alpha, dim=0)
        return probs

    @property
    def chosen_index(self):
        # return the index and operation with the max prob
        probs = self.probs_over_ops.data.cpu().numpy()
        index = int(np.argmax(probs))
        return index, probs[index]

    @property
    def chosen_op(self):
        # return chosen operation according to the chosen index
        index, _ = self.chosen_index
        return self.candidate_ops[index]

    @property
    def random_op(self):
        # random select one path
        index = np.random.choice(
            [_i for _i in range(self.n_choices)], 1
        )[0]
        return self.candidate_ops[index]

    def entropy(self, eps=1e-8):
        probs = self.probs_over_ops
        log_probs = torch.log(probs + eps)
        entropy = -torch.sum(torch.mul(probs, log_probs))
        return entropy

    def is_zero_layer(self):
        return self.active_op.is_zero_layer()

    @property
    def active_op(self):
        return self.candidate_ops[self.active_index[0]]

    def set_chosen_op_active(self):
        chosen_idx, _ = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.n_choices)]

    def forward(self, x):
        #print(MixedEdge.MODE)
        if MixedEdge.MODE == 'full' or MixedEdge == 'two':
            output = 0
            for _i in self.active_index:
                output_i = self.candidate_ops[_i](x)
                output = output + self.AP_path_wb[_i] * output_i
            for _i in self.inactive_index:
                output_i = self.candidate_ops[_i](x)
                output = output + self.AP_path_wb[_i] * output_i.detach()
        elif MixedEdge.MODE == 'full_v2':
            # when update architecture parameter
            def run_function(candidate_ops, active_id):
                def forward(_x):
                    return candidate_ops[active_id](_x)
                return forward
            '''
                :notes:
                the gradient of arch_param need to calculate gradient of all the binary_gates
                /sum_j^N /partial L / /partial gate\_grad_j
                each binary_gates grad need output_i and grad_output
            '''
            # calculate binary_grads in backward pass
            def backward_function(candidate_ops, active_id, binary_gates):
                def backward(_x, _output, grad_output):
                    binary_grads = torch.zeros_like(binary_gates.data)
                    with torch.no_grad():
                        for k in range(len(candidate_ops)):
                            if k != active_id:
                                  out_k = candidate_ops[k](_x.data)
                            else:
                                  out_k = _output.data
                            grad_k = torch.sum(out_k * grad_output)
                            binary_grads[k] = grad_k
                    return binary_grads
                return backward
            # self.active_index is a array, pick item()
            output = ArchGradientFunction.apply(
                x, self.AP_path_wb, run_function(self.candidate_ops, self.active_index[0]),
                backward_function(self.candidate_ops, self.active_index[0], self.AP_path_wb)
            )
            # self.AP_path_wb requires_grad=True, have value but grad is None
            #print(self.AP_path_wb.grad)
        else:
            # when training
            output = self.active_op(x)
        return output

    def module_str(self):
        # return the chosen operation
        chosen_index, probs = self.chosen_index
        return 'MixedEdge({}, {:.3f})'.format(self.candidate_ops[chosen_index].module_str, probs)

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    def get_flops(self, x):
        # get flops of active paths
        flops = 0
        for i in self.active_index:
            delta_flop, _ = self.candidate_ops[i].get_flops(x)
            flops += delta_flop
        return flops, self.forward(x)

    def binarize(self):
        # print('hi im here')
        # active_index,
        # inactive_index,
        # AP_path_wb,
        # log_prob(optional)
        # current_probs_over_ops(optional)
        self.log_prob = None
        # reset binary gates
        self.AP_path_wb.data.zero_()
        # binarize according to probs
        probs = self.probs_over_ops
        if MixedEdge.MODE == 'two':
            # sample two ops according to 'probs'
            sample_ops = torch.multinomial(probs.data, 2, replacement=False)
            probs_slice = F.softmax(torch.stack([
                self.AP_path_alpha[index] for index in sample_ops
            ]), dim=0)
            self.current_prob_over_ops = torch.zeros_like(probs)
            for i, index in enumerate(sample_ops):
                # only for the two chosen path, others are zero
                self.current_prob_over_ops[index] = probs_slice[i]
            # chose one to be active and other to be inactive
            c = torch.multinomial(probs_slice.data, 1)[0]
            active_op = sample_ops[c].item()
            inactive_op = sample_ops[1-c].item()

            # TODO: here setting active_index and inactive_index
            # will affect some process that related to active_index and inactive_index
            self.active_index = [active_op]
            self.inactive_index = [inactive_op]
            # set binary_gate
            self.AP_path_wb.data[active_op] = 1.0
        else:
            sample = torch.multinomial(probs.data, 1)[0].item()
            # when not 'two' mode unused_modules is None, involved_index = active_index + inactive_index
            self.active_index = [sample]

            self.inactive_index = [_i for _i in range(0, sample)] + \
                                  [_i for _i in range(sample+1, self.n_choices)]

            self.log_prob = torch.log(probs[sample])
            self.current_prob_over_ops = probs

            self.AP_path_wb.data[sample] = 1.0

        # TODO: avoid over-regularization
        for _i in range(self.n_choices):
            for name, param in self.candidate_ops[_i].named_parameters():
                param.grad = None

    def set_arch_param_grad(self):

        '''
        # where comes from
        if self.AP_path_wb.grad is None :
            print(self.AP_path_wb)
            print(self.active_op)
            #print(self.module_str)
            '''
        # TODO: change line of binary_grads = self.AP_path_wb.grad.data

        if self.active_op.is_zero_layer():
            self.AP_path_alpha.grad = None
            return

        binary_grads = self.AP_path_wb.grad.data

        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)

        if MixedEdge.MODE == 'two':
            involved_idx = self.active_index + self.inactive_index
            # probs are reset accordingly
            probs_slice = F.softmax(torch.stack([
                self.AP_path_alpha[index] for index in involved_idx
            ]), dim=0).data
            for i in range(2):
                for j in range(2):
                    origin_i = involved_idx[i] # original_index
                    origin_j = involved_idx[j]
                    self.AP_path_alpha.grad.data[origin_i] += \
                        binary_grads[origin_j] * probs_slice[j] * (delta_ij(i, j) - probs_slice[i])
            # set tuple, index, alpha
            for _i, index in enumerate(self.active_index):
                self.active_index[_i] = (index, self.AP_path_alpha.data[index].item())
            for _i, index in enumerate(self.inactive_index):
                self.inactive_index[_i] = (index, self.AP_path_alpha.data[index].item())
        else:
            probs = self.probs_over_ops.data
            for i in range(self.n_choices):
                for j in range(self.n_choices):
                    self.AP_path_alpha.grad.data[i] += binary_grads[j] * probs[j] * (delta_ij(i, j) - probs[i])
        return

    def rescale_updated_arch_param(self):
        # only in MixedEdge.MODE == two
        if not isinstance(self.active_index[0], tuple):
            assert self.active_op.is_zero_layer()
            return
        involved_index = [index for index, _ in (self.active_index + self.inactive_index)]
        old_alphas = [alpha for _, alpha in (self.active_index + self.inactive_index)]
        new_alphas = [self.AP_path_alpha.data[index] for index in involved_index]

        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) / sum([math.exp(alpha) for alpha in old_alphas])
        )
        for index in involved_index:
            self.AP_path_alpha.data[index] -= offset

class ArchGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        detached_x = detach_variable(x) # detach from computational graph but still requires_grad

        with torch.enable_grad():
            output = run_func(detached_x)
        ctx.save_for_backward(detached_x, output)
        return output.data
    @staticmethod
    def backward(ctx, grad_outputs):
        detached_x, output = ctx.saved_tensors
        grad_x = torch.autograd.grad(output, detached_x, grad_outputs, only_inputs=True)

        # compute gradient w.r.t. binary_gates
        binary_grads = ctx.backward_func(detached_x.data, output.data, grad_outputs.data)
        #print(binary_grads)
        # return value related to forward arguments
        return grad_x[0], binary_grads, None, None