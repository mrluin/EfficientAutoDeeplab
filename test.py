import torch
import torch.nn as nn
import torch.nn.functional as F
import sys





'''
torch.autograd.set_detect_anomaly(True)
arch_parameters = torch.randn((2, 3), requires_grad=True)
gumbels = -torch.empty_like(arch_parameters).exponential_().log()
logits = torch.zeros_like(arch_parameters)

probs = torch.zeros_like(arch_parameters)

#logits = arch_parameters.log_softmax(dim=-1) + gumbels

for i in range(2):
    logits[i][1:] = arch_parameters[i][1:].log_softmax(dim=-1) + gumbels[i][1:]
    #logits[i][1:] = arch_parameters[i][1:].log_softmax(dim=-1) + gumbels[i][1:]

    probs[i][1:] = F.softmax(logits.clone()[i][1:], dim=-1)


index = probs.max(-1, keepdim=True)[1]
one_h = torch.zeros_like(logits).scatter_(-1, index, 1.0)

hardwts = one_h - probs.detach() + probs
print('hardwts:', hardwts.requires_grad)

value = torch.randn((2, 3), requires_grad=True)
result = sum(sum(hardwts * value))
print(arch_parameters.grad)
result.backward()

print(hardwts.grad)
print(arch_parameters.grad)
'''
'''
actual_path = [1,2,3,4,5,6,7,788]
aa = [(1,2), [(3,4)]]

file = open('./test.log', 'w')
for i in range(10):
    file.write(str(aa))
    file.flush()
'''
'''
import json
import os
from configs.train_search_config import obtain_train_search_args
def save_configs(configs, save_path, phase):

    if configs is not None:
        config_path = os.path.join(save_path, '{:}.config'.format(phase))
        print('=' * 30 + '\n' + 'Run Configs dumps to {}'.format(config_path))
        json.dump(configs, open(config_path, 'w'), indent=4)

args = obtain_train_search_args()


args.__dict__['path'] = '123'
print(args.path)
#save_configs(args.__dict__, './', 'test')

with open('./test.config', 'r') as f:
    load_dict = json.load(f)

print(load_dict['path'])
'''
print(len([j for i in range(1) for j in range(i+2)]))
print([j for i in range(1) for j in range(i+2)])