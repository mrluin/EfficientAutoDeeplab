import logging
import time
from datetime import timedelta
import torch
import torch.nn as nn



model = nn.Linear(10, 10, False)

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
for param_group in optimizer.param_groups:
    param_group['lr'] = 0.05
    print(param_group['lr'])
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)


for epoch in range(10):
    lr_scheduler.step(epoch)
    print(lr_scheduler.get_lr())

print(optimizer)






