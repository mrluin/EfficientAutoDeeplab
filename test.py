import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import json
import torchvision.datasets as datasets
import numpy as np
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
'''
print(len([j for i in range(1) for j in range(i+2)]))
print([j for i in range(1) for j in range(i+2)])
'''

'''
checkpoint = torch.load('./seed-21054-search.pth')
print(checkpoint['state_dict'])
'''
'''
f = open('./warm40-epochs200-freq15-wlr006-slr0025-alr0004-bs16-entropy-search.json', 'r')
dicct = json.load(f)
f.close()
print(dicct['jsons'].keys())
print(dicct['jsons']['window_3806cad6e16850'].keys())
print(dicct['jsons']['window_3806cad6e16850']['content']['data'][0]['x']) # train_miou
print(dicct['jsons']['window_3806cad6e16850']['content']['data'][0]['y']) # train_miou

print(dicct['jsons']['window_3806cad6e16850']['content']['data'][1]['x']) # valid_miou
print(dicct['jsons']['window_3806cad6e16850']['content']['data'][1]['y']) # valid_miou

x = dicct['jsons']['window_3806cad6e16850']['content']['data'][0]['x']
y = dicct['jsons']['window_3806cad6e16850']['content']['data'][0]['y']
'''
'''
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(size=2, color='black'),
    name='Sine'
))
'''

'''
fig.add_trace(go.Scatter(
    x=x,
    y=y_noise,
    mode='markers',
    marker=dict(
        size=6,
        color='royalblue',
        symbol='circle-open'
    ),
    name='Noisy Sine'
))

'''
'''
fig.add_trace(go.Scatter(
    x=x,
    y=signal.savgol_filter(y,
                           53, # window size used for filtering
                           3), # order of fitted polynomial
    mode='markers',
    marker=dict(
        size=6,
        color='mediumpurple',
        symbol='triangle-up'
    ),
    name='Savitzky-Golay'
))
'''
#fig.show()






'''
yhat = savgol_filter(y, 21, 3) # window size 51, polynomial order 3

#yhat = savitzky_golay(y, 51, 3) # window size 51, polynomial order 3
plt.title('test for smoothing')
plt.xlabel('epochs')
plt.ylabel('unknown')
#plt.tick_params(axis='both', which='major', labelsize=14)
plt.plot(x,y, label='unsmooth', linewidth=1.5, linestyle='-', color='green', alpha=0.2)  # unsmooth
plt.plot(x,yhat, label='smooth', linewidth=1.5, linestyle='-', color='red', alpha=1)     # smooth
#plt.fill_between(x, y, yhat, color='green', alpha=0.2)
plt.legend(loc='upper right')
# ax为两条坐标轴的实例
ax = plt.gca()
# x轴的刻度间隔设置为20，并存储在变量中

x_major_locator = MultipleLocator(20)
# 设置水平坐标间隔
ax.xaxis.set_major_locator(x_major_locator)
# 设置最大值以及最小值
plt.xlim(-0.5, 200)
plt.grid(color='grey', linestyle=':')
plt.xscale('linear')
plt.yscale('linear')
plt.autoscale()
plt.show()
'''
# import os
# import glob
# import shutil
#
# ''' dirname and basename'''
# def get_file_list(path, script_to_save):
#     # path to project root path
#     file_list = []
#     def rt_files(file_path):
#         if not os.path.isdir(file_path):
#             file_list.append(file_path)
#         else:
#             for file in glob.glob(os.path.join(file_path, '*')):
#                 rt_files(file)
#     rt_files(script_to_save)
#
#     print(file_list)
#
#     os.mkdir(os.path.join(path, 'scripts'))
#     for script in file_list:
#         dst_file = os.path.join(path, 'scripts', os.path.basename(script))
#         shutil.copy(script, dst_file)
#     #return file_list
#
# #print(len(get_file_list('./', 'D:/Efficient_AutoDeeplab/')))
# get_file_list('./', 'D:/Efficient_AutoDeeplab/')
#
# shutil.copytree('../Efficient_AutoDeeplab', 'D:/Efficient_AutoDeeplab/tree_scripts')
#
# ''' the use of shutil.copy and shutil.copytree '''


from models.normal_models.Unet_ji import Unet_ji
from models.normal_models.enet import ENet
from models.normal_models.erfnet import ERFNet
from models.normal_models.esfnet import ESFNet
from models.normal_models.FCN_ji import FCN_ji
from utils.flop_benchmark import get_model_infos
'''
model = Unet_ji(2)
flop, param = get_model_infos(model, [1, 3, 512, 512])
print('Unet:: FLOPS & PARAMS ')
print('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
'''
'''
model = ENet(2)
flop, param = get_model_infos(model, [1, 3, 512, 512])
print('ENet:: FLOPS & PARAMS ')
print('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
model = ERFNet(2)
flop, param = get_model_infos(model, [1, 3, 512, 512])
print('ERFNet:: FLOPS & PARAMS')
print('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
model = ESFNet(2, interpolate=True)
flop, param = get_model_infos(model, [1, 3, 512, 512])
print('ESFNet_interpolate:: FLOPS & PARAMS')
print('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
model = ESFNet(2, interpolate=False)
flop, param = get_model_infos(model, [1, 3, 512, 512])
print('ESFNet_not_interpolate:: FLOPS & PARAMS')
print('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
'''
'''
model = FCN_ji(2)
flop, param = get_model_infos(model, [1, 3, 512, 512])
print('FCN:: FLOPS & PARAMS ')
print('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
'''
# check convergence epochs for the best architecture
'''
checkpoint_path = '/home/linjingbo/Jingbo.TTB/Workspace/final_exp/20-100-005-0025-0003-0006003-1-linear-random-trial2-search/06-Jan-at-07-04-49/checkpoints/seed-6098-search-best.pth'
arch_checkpoint_path = '/home/linjingbo/Jingbo.TTB/Workspace/final_exp/20-100-005-0025-0003-0006003-1-linear-random-trial2-search/06-Jan-at-07-04-49/checkpoints/seed-6098-arch-best.pth'
checkpoint = torch.load(checkpoint_path)
arch_checkpoint = torch.load(arch_checkpoint_path)
best_epoch = checkpoint['start_epochs'] - 1
actual_path = arch_checkpoint['actual_path']

print('best_epochs::', best_epoch)
print('actual_path::', actual_path)

'''  
                              #start_epochs-1
# window_380f3118a1c92c 001005       90
# window_380f5c9177f3f6 001005linear 94
# window_380dccfff296c8 001001       90
# window_380dcd01f59fa8 00010001     95
# window_380ff39b09453c warm-noentropy-nop 96 [0 1 0 0 1 0 1 0 0 0 1 1]     92 96
# window_380c721b726022 warm-p-noentropy   93 [0 0 0 0 1 1 2 3 3 3 2 1]
# window_3810383cc99800 00010005     96
# window_3810f47496689a 0005001linear97
# window_3810e671ea130e nude         99       actual_path:: [1 0 0 1 1 2 3 2 2 2 2 1] 99
# window_3811b2f61a4908 0008004linear95
# window_3811bb5b40fef8 0006003linear94
# window_381280bbb84c80 0006003linear-trial1 99
# window_381280c2da1580 0006003linear-trial2 95
f = open('20-100-005-0025-0003-0006003-1-linear-random-trial2-search.json')
json_dict = json.load(f)
f.close()

#print(json_dict['jsons'].keys())
for key in json_dict['jsons'].keys():
    if json_dict['jsons'][key]['title'] == 'entropy':
        print(key)

# constraint initial entropy 50.96009063720703
# without constraint initial entropy 53.98834228515625

print(json_dict['jsons']['window_381280c2da1580']['content']['data'][0]['x'])
print(json_dict['jsons']['window_381280c2da1580']['content']['data'][0]['y'])
print(50.96009063720703 - json_dict['jsons']['window_381280c2da1580']['content']['data'][0]['y'][94])

# visdom_log records 1-99 except the initial one, so when calculate \Delta S, need use best_epoch-1


# 001001 15.95 2.004


# 001005 19.54

#% 0006003 trial0 7.87 94 optimal 37 converge random_seed = 1
#% 0006003 trial1 10.07 99 optimal 97 converge
#% 0006003 trial2 11.79 95 optimal 93 converge
