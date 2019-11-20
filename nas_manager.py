import math
import torch
import os
import time
import logging
import json

from models.split_fabric_auto_deeplab import SplitFabricAutoDeepLab
from run_manager import *
from utils.common import set_manual_seed
from utils.common import AverageMeter
from utils.common import get_monitor_metric
from utils.metrics import Evaluator
from modules.mixed_op import MixedEdge
from utils.common import count_parameters
from utils.common import get_list_index
from models.new_model import NewNetwork

'''
# ArchSearchConfig: architecture parameter init and optimizer
# GradientArchSearchConfig: architecture parameter update related information, update scheduler
# ArchSearchRunManager: perform network training and architecture parameter training.
'''

class ArchSearchConfig:
    def __init__(self,
                 arch_init_type, arch_init_ratio, arch_optim_type,
                 arch_lr, arch_optim_params, arch_weight_decay,
                 target_hardware, ref_value):

        self.arch_init_type = arch_init_type
        self.arch_init_ratio = arch_init_ratio
        self.arch_optim_type = arch_optim_type
        self.arch_lr = arch_lr
        self.arch_optim_params = arch_optim_params
        self.arch_weight_decay = arch_weight_decay

        # TODO: related computational constraints
        self.target_hardware = target_hardware
        self.ref_value = ref_value

    @property
    def config(self):
        config = {
            'type' : type(self),
        }
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def get_update_schedule(self, iter_per_epoch):
        raise NotImplementedError

    def build_optimizer(self, params):
        # build arch_parameter optimizer
        # params: arch_parameters
        if self.arch_optim_type == 'adam':
            return torch.optim.Adam(
                params, self.arch_lr, weight_decay=self.arch_weight_decay, **self.arch_optim_params
            )
        else:
            raise ValueError('do not support otherwise torch.optim.Adam')

class GradientArchSearchConfig(ArchSearchConfig):
    def __init__(self,
                 arch_init_type, arch_init_ratio, arch_optim_type,
                 arch_lr, arch_optim_params, arch_weight_decay,
                 target_hardware, ref_value,
                 grad_update_arch_param_every, grad_update_steps, grad_binary_mode, grad_data_batch,
                 grad_reg_loss_type, grad_reg_loss_params, **kwargs):
        super(GradientArchSearchConfig, self).__init__(
            arch_init_type, arch_init_ratio, arch_optim_type,
            arch_lr, arch_optim_params, arch_weight_decay,
            target_hardware, ref_value,
        )

        self.grad_update_arch_param_every = grad_update_arch_param_every # how often updates architecture parameter, per iteration
        self.grad_update_steps = grad_update_steps # how many steps updates architecture parameter within an update process
        self.grad_binary_mode = grad_binary_mode # full full_v2 two
        self.grad_data_batch = grad_data_batch

        self.grad_reg_loss_type = grad_reg_loss_type
        self.grad_reg_loss_params = grad_reg_loss_params

    def get_update_schedule(self, iter_per_epoch):
        schedule = {}
        for i in range(iter_per_epoch):
            if (i+1) % self.grad_update_arch_param_every == 0:
                schedule[i] = self.grad_update_steps # iteration i update arch_param self.grad_update_steps times.
        return schedule

    def add_regularization_loss(self, ce_loss, expected_value=None):
        # TODO: need confirm, expected_value related to latency costraint
        # do not use expected_value, latency constraint
        if expected_value is None:
            return ce_loss

        if self.grad_reg_loss_type == 'mul#log ':
            alpha = self.grad_reg_loss_params.get('alpha', 1)
            beta = self.grad_reg_loss_params.get('beta', 0.6)
            reg_loss = (torch.log(expected_value) / math.log(self.ref_value)) ** beta
            return alpha * ce_loss * reg_loss
        elif self.grad_reg_loss_type == 'add#linear':
            reg_lambda = self.grad_reg_loss_params.get('lambda', 2e-1)
            reg_loss = reg_lambda * (expected_value - self.ref_value) / self.ref_value
            return ce_loss + reg_loss
        elif self.grad_reg_loss_type is None:
            return ce_loss
        else:
            raise ValueError('do not support {}'.format(self.grad_reg_loss_type))

class ArchSearchRunManager:
    def __init__(self,
                 path, super_net,
                 run_config: RunConfig,
                 arch_search_config: ArchSearchConfig):

        self.run_manager = RunManager(path, super_net, run_config, out_log=True, measure_latency=None)
        self.arch_search_config = arch_search_config

        # init architecture parameters
        # TODO: model init parameters
        self.net.init_arch_params(
            self.arch_search_config.arch_init_type,
            self.arch_search_config.arch_init_ratio
        )

        # build architecture optimizer
        self.arch_optimizer = self.arch_search_config.build_optimizer(self.net.architecture_parameters())
        #self.arch_optimizer = self.arch_optimizer.to(self.run_manager.device)

        self.warmup = True # should warmup or not
        self.warmup_epoch = 0 # current warmup epoch

    @property
    def net(self):
        #return self.run_manager.net.module
        return self.run_manager.net
    '''# use combination method in run_manager 
    def write_log(self, log_str, prefix, should_print=True):
        # related to arch_search log, excluding train valid and test statistics
        log_file = os.path.join(self.run_manager.log_path, prefix+'.txt')
        with open(log_file, 'a') as fout:
            fout.write(log_str + '\n')
            fout.flush()
        if should_print:
            logging.info(log_str)
            '''
    def load_model(self, ckptfile_path=None):

        assert ckptfile_path is not None and os.path.exists(ckptfile_path), \
            'ckptfile can not be found'
        try:
            '''
            if ckpt_filename is None or not os.path.exists(ckpt_filename):
                # TODO: this case has issue
                ckpt_filename = '{}/checkpoint.pth.tar'.format(self.save_path)
                '''
            if self.run_manager.out_log:
                logging.info('Loading Checkpoint {}'.format(ckptfile_path))

            if torch.cuda.is_available():
                checkpoint = torch.load(ckptfile_path)
            else:
                checkpoint = torch.load(ckptfile_path, map_location='cpu')

            model_dict = self.net.state_dict()
            model_dict.update(checkpoint['state_dict'])
            self.net.load_state_dict(model_dict)

            # TODO:  why set new manual seed
            new_manual_seed = int(time.time())
            set_manual_seed(new_manual_seed)

            # other elements
            if 'epoch' in checkpoint:
                self.start_epoch = checkpoint['epoch'] + 1
            if 'best_{}'.format(self.run_manager.monitor_metric) in checkpoint:
                self.best_monitor = checkpoint['best_{}'.format(self.run_manager.monitor_metric)]
            # optimizer for only training, weight_optimizer and arch_optimizer for train_search phrase
            if 'optimizer' in checkpoint:
                self.run_manager.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'weight_optimizer' in checkpoint:
                self.run_manager.optimizer.load_state_dict(checkpoint['weight_optimizer'])
            if 'arch_optimizer' in checkpoint:
                self.arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
            if 'warmup' in checkpoint:
                self.warmup = checkpoint['warmup']
            if self.warmup and 'warmup_epoch' in checkpoint:
                self.warmup_epoch = checkpoint['warmup_epoch'] + 1
            if self.run_manager.out_log:
                logging.info('Loaded Checkpoint {}'.format(ckptfile_path))
        except Exception:
            if self.run_manager.out_log:
                logging.info('Fail to load Checkpoint {}'.format(ckptfile_path))

    ''' training related methods '''
    def validate(self):
        # for validation phrase, not train search phrase, only performing validation
        # valid_loader batch_size = test_batch_size
        # have already equals to test_batch_size in DataProvider
        self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.run_manager.run_config.train_batch_size
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = False


        #print('before validate net active_index')
        #print(self.net.active_index)
        self.net.set_chosen_op_active()
        #print('validate net active_index')
        #print(self.net.active_index)

        self.net.unused_modules_off()
        # TODO: network_level and cell_level decode in net_flops() method
        # TODO: valid on validation set (from training set) under train mode
        # only have effect on operation related to train mode
        # like bn or dropout
        loss, acc, miou, fscore = self.run_manager.validate(is_test=False, net=self.net, use_train_mode=True)
        flops = self.run_manager.net_flops()
        params = count_parameters(self.net)
        # target_hardware is None by default
        if self.arch_search_config.target_hardware in ['flops', None]:
            latency = 0
        else:
            raise NotImplementedError

        self.net.unused_modules_back()
        return loss, acc, miou, fscore, flops, params

    def warm_up(self, warmup_epochs):
        if warmup_epochs <=0 :
            print('warmup close')
            return

        lr_max = 0.05
        data_loader = self.run_manager.run_config.train_loader
        iter_per_epoch = len(data_loader)
        total_iteration = warmup_epochs * iter_per_epoch

        print('warmup begin')
        '''
        # check params device
        for name, param in self.net.named_parameters():
            device = param.device
            if device == torch.device('cpu'):
                print('ERROR in', name, device)
                '''
        '''
        def hook(module, grad_input, grad_output):
            for i in range(len(grad_input)):
                if grad_input[i] is not None and torch.Tensor.type(grad_input[i]) == 'torch.FloatTensor':
                    print('yes')
            for j in range(len(grad_output)):
                if grad_output[j] is not None and torch.Tensor.type(grad_output[j]) == 'torch.FloatTensor':
                    print('yes')
        def forward_hook(module, input, output):
            for i in range(len(input)):
                if input[i] is not None and torch.Tensor.type(input[i]) != 'torch.cuda.FloatTensor':
                    print('find the error input')
            for j in range(len(output)):
                if output[j] is not None and torch.Tensor.type(output[j]) != 'torch.cuda.FloatTensor':
                    print('find the error output')
        # for module in self.net.modules():
            # module.register_backward_hook(hook)
            # module.register_forward_hook(forward_hook)
            '''
        for epoch in range(self.warmup_epoch, warmup_epochs):
            print('\n', '-'*30, 'Warmup Epoch: {}'.format(epoch+1), '-'*30, '\n')
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accs = AverageMeter()
            mious = AverageMeter()
            fscores = AverageMeter()

            self.run_manager.net.train()
            print('\twarm up epoch {}'.format(epoch))
            end = time.time()
            for i, (datas, targets) in enumerate(data_loader):

                # TODO: evaluate memory allocation
                if i == 1 : break

                #print('before datas')
                #print('memory_allocated', torch.cuda.memory_allocated())
                #print('max_memory_allocated', torch.cuda.max_memory_allocated())

                if torch.cuda.is_available():
                    datas = datas.to(self.run_manager.device, non_blocking=True)
                    targets = targets.to(self.run_manager.device, non_blocking=True)
                else:
                    raise ValueError('do not support cpu version')
                data_time.update(time.time()-end)

                #print('after datas')
                #print('memory_allocated', torch.cuda.memory_allocated())
                #print('max_memory_allocated', torch.cuda.max_memory_allocated())


                # adjust warmup_lr
                warmup_lr = self.run_manager.run_config.adjust_learning_rate(
                    self.run_manager.optimizer, epoch, i, iter_per_epoch, lr_max, warmup_lr=True
                )
                #current_iteration = epoch * iter_per_epoch + i
                #warmup_lr = 0.5 * lr_max * (1 + math.cos(math.pi * current_iteration / total_iteration))
                #for param_group in self.run_manager.optimizer.param_groups:
                #    param_group['lr'] = warmup_lr

                self.net.reset_binary_gates()
                #print('warm_up net active_index')
                #print(self.net.active_index)
                self.net.unused_modules_off()

                #print('after binarize')
                #print('memory_allocated', torch.cuda.memory_allocated())
                #print('max_memory_allocated', torch.cuda.max_memory_allocated())

                logits = self.net(datas)

                #print('after forward')
                #print('memory_allocated', torch.cuda.memory_allocated())
                #print('max_memory_allocated', torch.cuda.max_memory_allocated())

                '''
                if self.run_manager.run_config.label_smoothing > 0.:
                    raise NotImplementedError
                else:
                    loss = self.run_manager.criterion(logits, targets)
                    '''
                '''
                # check param device
                for name, param in self.net.named_parameters():
                    if torch.Tensor.type(param) == 'torch.FloatTensor':
                        print('Error in:', name)
                        '''
                loss = self.run_manager.criterion(logits, targets)
                #print('loss:', torch.Tensor.type(loss))
                self.net.zero_grad()
                '''
                # check param device
                for name, param in self.net.named_parameters():
                    if torch.Tensor.type(param) == 'torch.FloatTensor':
                        print('Error in:', name)
                        '''

                loss.backward()
                #print('after backwards')
                #print('memory_allocated', torch.cuda.memory_allocated())
                #print('max_memory_allocated', torch.cuda.max_memory_allocated())

                self.run_manager.optimizer.step()
                self.net.unused_modules_back()

                # measure metrics and update

                evaluator = Evaluator(self.run_manager.run_config.nb_classes)
                evaluator.add_batch(targets, logits)
                acc = evaluator.Pixel_Accuracy()
                miou = evaluator.Mean_Intersection_over_Union()
                fscore = evaluator.Fx_Score()
            
                losses.update(loss.data.item(), datas.size(0))
                accs.update(acc, datas.size(0))
                mious.update(miou, datas.size(0))
                fscores.update(fscore, datas.size(0))


                # gradient and update parameters
                # self.run_manager.optimizer.zero_grad()
                # TODO: here should zero weight_param, arch_param, and binary_param
                # self.run_manager.net.zero_grad()

                batch_time.update(time.time()-end)
                end = time.time()

                if i % self.run_manager.run_config.print_freq == 0 or i + 1 == iter_per_epoch:
                    batch_log = 'Warmup Train\t[{0}][{1}/{2}]\tlr\t{lr:.5f}\n' \
                                'Time\t{batch_time.val:.3f}\t({batch_time.avg:.3f})\n' \
                                'Data\t{data_time.val:.3f}\t({data_time.avg:.3f})\n' \
                                'Loss\t{losses.val:6.4f}\t({losses.avg:6.4f})\n' \
                                'Acc\t{accs.val:6.4f}\t({accs.avg:6.4f})\n' \
                                'mIoU\t{mious.val:6.4f}\t({mious.avg:6.4f})\n' \
                                'F\t{fscores.val:6.4f}\t({fscores.avg:6.4f})\n'.format(
                        epoch+1, i, iter_per_epoch, lr=warmup_lr, batch_time=batch_time, data_time=data_time,
                        losses=losses, accs=accs, mious=mious, fscores=fscores
                    )
                    # TODO: do not use self.write_log
                    self.run_manager.write_log(batch_log, 'train')
                #break

            # TODO: in warm_up phase, does not update network_arch_parameters,
            # the super_net path used in validate could be invalid.
            # perform validate at the end of each epoch
            valid_loss, valid_acc, valid_miou, valid_fscore, flops, params = self.validate()
            valid_log = 'Warmup Valid\t[{0}/{1}]\tLoss\t{2:.6f}\tAcc\t{3:6.4f}\tMIoU\t{4:6.4f}\tF\t{5:6.4f}\tflops\t{6:}M\tparams{7:}M'\
                .format(epoch+1, warmup_epochs, valid_loss, valid_acc, valid_miou, valid_fscore, flops, params / 1e6)
            valid_log += 'Train\t[{0}/{1}]\tLoss\t{2:.6f}\tAcc\t{3:6.4f}\tMIoU\t{4:6.4f}\tFscore\t{5:6.4f}'


            # target_hardware is None by default
            if self.arch_search_config.target_hardware not in [None, 'flops']:
                raise NotImplementedError

            self.run_manager.write_log(valid_log, 'valid')

            # continue warmup phrase

            self.warmup = epoch + 1 < warmup_epochs

            # To save checkpoint in warmup phase at specific frequency.
            if (epoch + 1) % self.run_manager.run_config.save_ckpt_freq == 0:
                state_dict = self.net.state_dict()
                # TODO: why
                # rm architecture parameters & binary gates
                for key in list(state_dict.keys()):
                    if 'AP_path_alpha' in key or 'AP_path_wb' in key:
                        state_dict.pop(key)
                checkpoint = {
                    'state_dict': state_dict,
                    'warmup': self.warmup
                }
                if self.warmup:
                    checkpoint['warmup_epoch'] = epoch
                self.run_manager.save_model(epoch, checkpoint, is_best=False,
                                            checkpoint_file_name='checkpoint-warmup.pth.tar')


    def train(self, fix_net_weights=False):
        data_loader = self.run_manager.run_config.train_loader
        iter_per_epoch = len(data_loader)
        arch_update_flag = 0
        if fix_net_weights: # used to debug
            data_loader = [(0, 0)] * iter_per_epoch
            print('Train Phase close for debug')
            arch_update_flag = 0

        # get fabric_path_alpha and AP_path_alpha
        arch_param_num = len(list(self.net.architecture_parameters()))
        # get fabric_path_wb and AP_path_wb
        binary_gates_num = len(list(self.net.binary_gates()))
        # get network weight params
        weight_param_num = len(list(self.net.weight_parameters()))

        print(
            '#arch_params: {}\t #binary_gates: {}\t # weight_params: {}'.format(
                arch_param_num, binary_gates_num, weight_param_num))

        update_schedule = self.arch_search_config.get_update_schedule(iter_per_epoch)

        # TODO: print arch_param update_schedule
        #print(update_schedule)

        for epoch in range(self.run_manager.start_epoch, self.run_manager.run_config.total_epochs):
            print('\n', '-'*30, 'Train Epoch: {}'.format(epoch+1), '-'*30, '\n')

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accs = AverageMeter()
            mious = AverageMeter()
            fscores = AverageMeter()

            #self.run_manager.net.train()
            self.net.train()

            end = time.time()
            for i, (datas, targets) in enumerate(data_loader):
                #lr
                lr = self.run_manager.run_config.adjust_learning_rate(
                    self.run_manager.optimizer, epoch, i, iter_per_epoch
                )
                # TODO: network entropy

                # train network weight parameter if not fix_net_weights

                if not fix_net_weights:
                    if torch.cuda.is_available():
                        datas = datas.to(self.run_manager.device, non_blocking=True)
                        targets = targets.to(self.run_manager.device, non_blocking=True)
                    else:
                        raise ValueError('do not support cpu version')

                    data_time.update(time.time() - end)

                    # compute output
                    self.net.reset_binary_gates()
                    self.net.unused_modules_off()
                    #print(self.net._unused_modules)
                    #logits = self.run_manager.net(datas)
                    logits = self.net(datas)
                    # loss
                    if self.run_manager.run_config.label_smoothing > 0.:
                        raise NotImplementedError
                    else:
                        loss = self.run_manager.criterion(logits, targets)
                    # metrics and update
                    evaluator = Evaluator(self.run_manager.run_config.nb_classes)
                    evaluator.add_batch(targets, logits)
                    acc = evaluator.Pixel_Accuracy()
                    miou = evaluator.Mean_Intersection_over_Union()
                    fscore = evaluator.Fx_Score()
                    losses.update(loss.data.item(), datas.size(0))
                    accs.update(acc.item(), datas.size(0))
                    mious.update(miou.item(), datas.size(0))
                    fscores.update(fscore.item(), datas.size(0))

                    # compute gradient and do SGD step
                    # zero out, network weight parameters, binary_gates, and arch_param
                    self.run_manager.net.zero_grad()
                    #self.net.zero_grad()
                    loss.backward()
                    # here only update network weight parameters
                    self.run_manager.optimizer.step()
                    self.net.unused_modules_back()

                    # TODO: confirm the correct place of batch_time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    # training log per print_freq
                    if i % self.run_manager.run_config.print_freq == 0 or i + 1 == iter_per_epoch:
                        batch_log = 'Train\t[{0}][{1}/{2}]\tlr {lr:.5f}\n' \
                                    'Time\t{batch_time.val:.3f}\t({batch_time.avg:.3f})\n' \
                                    'Data\t{data_time.val:.3f}\t({data_time.avg:.3f})\n' \
                                    'Loss\t{losses.val:6.4f}\t({losses.avg:6.4f})\n' \
                                    'Acc\t{accs.val:6.4f}\t({accs.avg:6.4f})\n' \
                                    'MIoU\t{mious.val:6.4f}\t({mious.avg:6.4f})\n' \
                                    'F\t{fscores.val:6.4f}\t({fscores.avg:6.4f})\n'.format(
                            epoch + 1, i, iter_per_epoch, lr=lr, batch_time=batch_time, data_time=data_time,
                            losses=losses, accs=accs, mious=mious, fscores=fscores
                        )
                        self.run_manager.write_log(batch_log, 'train', should_print=True)

                # TODO: skip arch_param update in the first epoch
                # epoch >= 0 used to debug in gradient_step
                if epoch >= 0:
                    if not fix_net_weights:
                        for j in range(update_schedule.get(i, 0)): # step i update arch_param times
                            start_time = time.time()
                            if isinstance(self.arch_search_config, GradientArchSearchConfig):
                                # target_hardware is None, exp_value is None by default
                                #print(self.net.inactive_index)
                                time_log, arch_loss, arch_acc, arch_miou, arch_fscore, exp_value = self.gradient_step()
                                if i % self.run_manager.run_config.print_arch_param_step_freq == 0 or i + 1 == iter_per_epoch:
                                    used_time = time.time() - start_time
                                    # current architecture information
                                    # performance and flops
                                    log_str = 'Architecture\t[{}-{}]\tTime\t{:.4f}\n' \
                                              'Loss\t{:.6f}\tAcc\t{:6.4f}\tMIoU\t{:6.4f}\tFscore\t{:6.4f}\n'\
                                        .format(epoch+1, i, used_time, arch_loss, arch_acc, arch_miou, arch_fscore)

                                    self.run_manager.write_log(log_str+time_log, prefix='gradient_search', should_print=True)
                            else:
                                raise ValueError('do not support version {}'.format(type(self.arch_search_config)))
                    else:
                        if arch_update_flag == 0:
                            for j in range(update_schedule.get(i, 0)):  # step i update arch_param times
                                start_time = time.time()
                                if isinstance(self.arch_search_config, GradientArchSearchConfig):
                                    # target_hardware is None, exp_value is None by default
                                    time_log, arch_loss, arch_acc, arch_miou, arch_fscore, exp_value = self.gradient_step()
                                    if i % self.run_manager.run_config.print_arch_param_step_freq == 0 or i + 1 == iter_per_epoch:
                                        used_time = time.time() - start_time
                                        # current architecture information
                                        # performance and flops
                                        log_str = 'Architecture\t[{}-{}]\tTime\t{:.4f}\n' \
                                                  'Loss\t{:.6f}\tAcc\t{:6.4f}\tMIoU\t{:6.4f}\tFscore\t{:6.4f}\n' \
                                            .format(epoch + 1, i, used_time, arch_loss, arch_acc, arch_miou,
                                                    arch_fscore)

                                        self.run_manager.write_log(log_str + time_log, prefix='gradient_search',
                                                                   should_print=True)
                                else:
                                    raise ValueError('do not support version {}'.format(type(self.arch_search_config)))
                            arch_update_flag = 1
                        else:
                            continue

            # print_save_arch_information
            if self.run_manager.run_config.print_save_arch_information:
                # print current network architecture at the end of each epoch
                self.run_manager.write_log('-'*30 + 'Current Architecture {}'.format(epoch+1)+ '-'*30+'\n', prefix='arch', should_print=True)
                log_str = self.net.module_str()
                self.run_manager.write_log(log_str, prefix='arch', should_print=True)
                self.run_manager.write_log('-' * 60, prefix='arch', should_print=True)

            # valid_freq
            if (epoch+1) % self.run_manager.run_config.validation_freq == 0:
                val_loss, val_acc, val_miou, val_fscore, flops, latency = self.validate()

                val_monitor_metric = get_monitor_metric(self.run_manager.monitor_metric, val_loss, val_acc, val_miou, val_fscore)
                self.run_manager.best_monitor = max(self.run_manager.best_monitor, val_monitor_metric)
                if not fix_net_weights:
                    val_log = 'Valid\t[{0}/{1}]\tLoss\t{2:6.4f}\tAcc\t{3:6.4f}\tMIoU\t{4:6.4f}\tFscore\t{5:6.4f}\n' \
                              'Train\tLoss{loss.avg:.6f}\tAcc{accs.avg:6.4f}\tMIoU{mious.avg:6.4f}\tFscore{fscores.avg:6.4f}\n'.format(
                        epoch+1, self.run_manager.run_config.total_epochs, val_loss, val_acc, val_miou, val_fscore,
                        loss=losses, accs=accs, mious=mious, fscores=fscores
                    )
                else:
                    val_log = 'Valid\t[{0}/{1}]\tLoss\t{2:6.4f}\tAcc\t{3:6.4f}\tMIoU\t{4:6.4f}\tFscore\t{5:6.4f}\n' \
                              .format(epoch + 1, self.run_manager.run_config.total_epochs, val_loss, val_acc, val_miou, val_fscore,)
                self.run_manager.write_log(val_log, 'valid', should_print=True)
            # save_ckpt_freq
            if (epoch+1) % self.run_manager.run_config.save_ckpt_freq == 0:
                self.run_manager.save_model(epoch, {
                    'warmup': False,
                    'epoch': epoch,
                    'weight_optimizer': self.run_manager.optimizer.state_dict(),
                    'arch_optimizer': self.arch_optimizer.state_dict(),
                    'state_dict': self.net.state_dict(),
                }, is_best=False, checkpoint_file_name=None)
        # after training phase
        if self.run_manager.run_config.save_normal_net_after_training:

            # get cell_arch_info and network_arch_info from genotype decode and viterbi_decode
            # construct cells and network, according to cell_arch_info and network_arch_info
            # obtain the whole network
            actual_path, network_space, gene = self.net.network_arch_cell_arch_decode()
            #print('\t-> actual_path:', actual_path)
            #print('\t-> network_space:', network_space)
            #print('\t-> gene:\n', gene)


            # obtain new_auto_deeplab
            normal_net = NewNetwork(actual_path, gene, self.run_manager.run_config.filter_multiplier,
                                    self.run_manager.run_config.block_multiplier, self.run_manager.run_config.steps,
                                    self.run_manager.run_config.nb_layers, self.run_manager.run_config.nb_classes,
                                    init_channels=128, conv_candidates=self.run_manager.run_config.conv_candidates)
            # TODO: device
            print('\t-> normal_net construct completely!')
            print('\t-> Total training params: {:.2f}'.format(count_parameters(normal_net) / 1e6))
            # TODO: fix issues in convert_to_normal_net()
            # TODO: network level need split cells
            # obtain normal cells
            # normal_net = self.net.cpu().convert_to_normal_net()

            # directory of network configs
            os.makedirs(os.path.join(self.run_manager.path, 'learned_net'), exist_ok=True)

            # TODO: get_network_config,
            # layer scale cell, selected edge_index, chosen operation.
            '''
            def get_network_arch(cell_arch, actual_path):
                config_log = 'Network-Level and Cell-Level Configs\n'
                config_log += 'Stem0\n Stem1\n'
                config_log += 'Layer0 Scale0 {}\n'.format('stem2')
                for i in range(self.run_manager.run_config.nb_layers):
                    next_scale = actual_path[i] # the next_scale
                    config_log += 'Layer{} Scale{} Cell:\n'.format(i+1, next_scale)
                    # TODO, get cell_index, related to cell type, split cell or proxy cell
                    cell_index = get_list_index(i, next_scale)
                    _cell_arch = cell_arch[cell_index] # [steps, chosen_edge_index, chosen_operation_index]
                    for x in _cell_arch:
                        config_log += '\tEdge {}, {}\n'.format(x[0], self.run_manager.run_config.conv_candidates[x[1]])

                last_scale = actual_path[-1]
                if last_scale == 0:
                    config_log += 'aspp4'
                elif last_scale == 1:
                    config_log += 'aspp8'
                elif last_scale == 2:
                    config_log += 'aspp16'
                elif last_scale == 3:
                    config_log += 'aspp32'
                else:
                    raise ValueError('invalid last_scale {}'.format(last_scale))

                return config_log
                '''

            # TODO: test config_log

            def get_network_config(normal_net, cell_arch, actual_path):
                nb_layers = 12
                config = {
                    'stem0': 'conv{}x{}_s{}_bn_relu'.format(normal_net.stem0.conv.kernel_size, normal_net.stem0.conv.kernel_size, normal_net.stem0.conv.stride),
                    'stem1': 'conv{}x{}_s{}_bn_relu'.format(normal_net.stem0.conv.kernel_size, normal_net.stem0.conv.kernel_size, normal_net.stem0.conv.stride),
                    'stem2': 'conv{}x{}_s{}_bn_relu'.format(normal_net.stem0.conv.kernel_size, normal_net.stem0.conv.kernel_size, normal_net.stem0.conv.stride),
                }
                for i in range(nb_layers):
                    next_scale = actual_path[i]
                    config['layer {}, scale {} cell'.format(i+1, next_scale)] = normal_net.cells[i].module_str()
                config['Aspp'] = normal_net.aspp

            config = get_network_config(normal_net, gene, actual_path)
            print('\t-> network config construct done')
            # done
            json.dump(config, open(os.path.join(self.run_manager.path, 'learned_net/net_config.txt'), 'w'), indent=4)
            # done
            json.dump(
                self.run_manager.run_config.config,
                open(os.path.join(self.run_manager.path, 'learned_net/run_config.txt'), 'w'),
                indent=4
            )
            # done
            torch.save(
                {'state_dict': normal_net.state_dict(),
                 'dataset': self.run_manager.run_config.dataset},
                os.path.join(self.run_manager.path, 'learned_net/init')
            )
            print('\t-> normal_network_config, run_config, ckpt saved done!')

    def gradient_step(self):

        def backward_hook(module, input, output):
            if (module.AP_path_wb.grad) is not None:
                print('hook not None')
            if module.AP_path_wb.grad is None:
                print('hook None')
            #print(module.AP_path_wb.requires_grad)


        assert isinstance(self.arch_search_config, GradientArchSearchConfig)

        # if data_batch is None, equals to train_batch_size
        if self.arch_search_config.grad_data_batch is None:
            self.run_manager.run_config.valid_loader.batch_sampler.batch_size = \
                self.run_manager.run_config.train_loader.batch_size
        else:
            self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.arch_search_config.grad_data_batch

        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = True

        #self.run_manager.net.train()
        self.run_manager.net.train()

        # TODO: MixedEdge !!!
        MixedEdge.MODE = self.arch_search_config.grad_binary_mode # full_v2
        SplitFabricAutoDeepLab.MODE = self.arch_search_config.grad_binary_mode
        #print('MixedEdge.MODE:',MixedEdge.MODE)
        time1 = time.time()

        # a batch of data from valid set (split from training)
        datas, targets = self.run_manager.run_config.valid_next_batch
        if torch.cuda.is_available():
            datas = datas.to(self.run_manager.device, non_blocking=True)
            targets = targets.to(self.run_manager.device, non_blocking=True)
        else:
            raise ValueError('do not support cpu version')
        time2 = time.time()

        #print('='*30)
        #print(self.net._unused_modules)
        self.net.reset_binary_gates()
        for module in self.net.redundant_modules:
            module.register_backward_hook(backward_hook)
        # print mixedop binarygates
        #print('mixed_operation binary gates, after binarize')
        #for param in self.net.cell_binary_gates():
        #    print(param.grad)

        self.net.unused_modules_off()


       # print(self.net._unused_modules)
        #print('forward:')
        print('\t before forward:')
        logits = self.run_manager.net(datas)
        print('\t after forward:')
        # print mixedop binarygates
        #print('mixed_operation binary gates, after forward')
        #for param in self.net.cell_binary_gates():
        #    print(param.grad)
        #logits = self.net(datas)
        time3 = time.time()
        # loss
        # TODO: why only simple ce_loss
        loss = self.run_manager.criterion(logits, targets)
        # metrics of batch of validation data
        evaluator = Evaluator(self.run_manager.run_config.nb_classes)
        evaluator.add_batch(targets, logits)
        acc = evaluator.Pixel_Accuracy()
        miou = evaluator.Mean_Intersection_over_Union()
        fscore = evaluator.Fx_Score()

        # target_hardware is None by default
        if self.arch_search_config.target_hardware is None:
            expected_value = None
        elif self.arch_search_config.target_hardware == 'mobile':
            raise NotImplementedError
        elif self.arch_search_config.target_hardware == 'flops':
            raise NotImplementedError
        else:
            raise NotImplementedError

        # if target_hardware is None, return simple ce_loss
        loss = self.arch_search_config.add_regularization_loss(loss, expected_value)

        # self.run_manager.net.zero_grad()
        # loss.backward() only binary gates have gradient
        # set arch_parameter gradients according to the gradients of binary gates
        #print('\tBefore self.net.zero_grad()')
        self.run_manager.net.zero_grad()
        #print('\tAfter self.net.zero_grad()')
        #print(self.net.binary_gates())
        #print('backward')
        print('\t before backward')
        loss.backward()
        print('\t after backward')
        #print('mixed_operation binary gates after backward')
        #for param in self.net.cell_binary_gates():
        #    print(param.grad)
        # print binary_gates.grad
        # but some mixed edge has no grad
        # for param in self.net.binary_gates():
        #    print(param.grad)

        # TODO: change mode
        MixedEdge.MODE = 'two'
        SplitFabricAutoDeepLab.MODE  = 'two'

        self.net.set_arch_param_grad() # get old_alphas

        self.arch_optimizer.step() # get new_alphas

        # TODO: change MODE
        if MixedEdge.MODE == 'two':
            self.net.rescale_updated_arch_param() # rescale updated arch_params according to old_alphas and new_alphas
        self.net.unused_modules_back()
        MixedEdge.MODE = None
        SplitFabricAutoDeepLab.MODE = None
        time4 = time.time()
        gradient_step_time_log = 'Data time: {:.4f},\tInference time: {:.4f},\tBackward time: {:.4f}\n'.format(time2-time1, time3-time2, time4-time3)
        # TODO: need modification
        return gradient_step_time_log, loss.data.item(), acc.item(), miou.item(), fscore.item(), expected_value.item() if expected_value is not None else None


