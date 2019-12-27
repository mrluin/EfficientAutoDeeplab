# ===============================
# author : Jingbo Lin
# contact: ljbxd180612@gmail.com
# github : github.com/mrluin
# ===============================
import math
import torch
import os
import time
import logging
import json

from exp.sufficient_update.run_manager import *

from utils.common import set_manual_seed
from utils.common import AverageMeter
from utils.common import get_monitor_metric
from utils.logger import save_checkpoint, time_string
from utils.metrics import Evaluator
from modules.mixed_op import MixedEdge
from utils.common import count_parameters
from utils.common import get_list_index
from models.new_model import NewNetwork

__all__ = ['ArchSearchConfig', 'ArchSearchRunManager']

class ArchSearchConfig:
    def __init__(self,
                 arch_init_type, arch_init_ratio,
                 warmup_lr,
                 arch_optimizer_type, arch_lr, arch_optimizer_params, arch_weight_decay,
                 tau_min, tau_max,
                 #target_hardware, ref_value,
                 arch_param_update_frequency, arch_param_update_steps, sample_arch_frequency,
                 reg_loss_type,
                 reg_loss_params, **kwargs):

        # optimizer
        self.arch_init_type = arch_init_type
        self.arch_init_ratio = arch_init_ratio
        self.arch_optimizer_type = arch_optimizer_type
        self.arch_lr = arch_lr
        self.arch_optimizer_params = arch_optimizer_params
        self.arch_weight_decay = arch_weight_decay
        self.warmup_lr = warmup_lr
        self.tau_min = tau_min
        self.tau_max = tau_max
        # update
        self.arch_param_update_frequency = arch_param_update_frequency
        self.arch_param_update_steps = arch_param_update_steps
        self.sample_arch_frequency = sample_arch_frequency
        # loss related to hardware contraint
        self.reg_loss_type = reg_loss_type
        self.reg_loss_params = reg_loss_params

        # TODO: related computational constraints
        # TODO: get rid of
        #self.target_hardware = target_hardware
        #self.ref_value = ref_value

    @property
    def config(self):
        config = {
            #'type' : type(self),
        }
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def get_update_schedule(self, iter_per_epoch):
        schedule = {}
        for i in range(iter_per_epoch):
            if (i+1) % self.sample_arch_frequency == 0:
                schedule[i] = self.arch_param_update_steps # iteration i update arch_param self.grad_update_steps times.
        return schedule


    def build_optimizer(self, params):
        if self.arch_optimizer_type == 'adam':
            return torch.optim.Adam(params, self.arch_lr, weight_decay=self.arch_weight_decay, **self.arch_optimizer_params)
        else: raise ValueError('do not support otherwise torch.optim.Adam')
    '''
    def add_regularization_loss(self, ce_loss, expected_value=None):
        # TODO: related hardware constrain, get rid of.
        if expected_value is None:
            return ce_loss

        if self.reg_loss_type == 'mul#log ':
            alpha = self.reg_loss_params.get('alpha', 1)
            beta = self.reg_loss_params.get('beta', 0.6)
            reg_loss = (torch.log(expected_value) / math.log(self.ref_value)) ** beta
            return alpha * ce_loss * reg_loss
        elif self.reg_loss_type == 'add#linear':
            reg_lambda = self.reg_loss_params.get('lambda', 2e-1)
            reg_loss = reg_lambda * (expected_value - self.ref_value) / self.ref_value
            return ce_loss + reg_loss
        elif self.reg_loss_type is None:
            return ce_loss
        else:
            raise ValueError('do not support {}'.format(self.reg_loss_type))
            '''

class ArchSearchRunManager:
    def __init__(self,
                 path, super_net,
                 run_config: RunConfig,
                 arch_search_config: ArchSearchConfig,
                 logger, vis=None):

        self.run_manager = RunManager(path, super_net, logger, run_config, out_log=True)
        self.arch_search_config = arch_search_config

        '''
        # arch_parameter init has implemented in SuperNetwork, performs initialization when construct the network.
        # init architecture parameters
        self.net.init_arch_params(
            self.arch_search_config.arch_init_type,
            self.arch_search_config.arch_init_ratio
        )
        '''

        # build architecture optimizer
        self.arch_optimizer = self.arch_search_config.build_optimizer(self.net.arch_parameters())
        self.warmup = True
        self.warmup_epoch = 0
        self.start_epoch = 0  # start_epoch, warmup_epoch, and total_epoch
        self.logger = logger
        self.vis = vis

        # for update arch_parameters


    @property
    def net(self):
        return self.run_manager.model

    def load_model(self, checkpoint_file=None):
        # only used in nas_manager
        assert checkpoint_file is not None and os.path.exists(checkpoint_file), \
            'checkpoint_file can not be found'

        if self.run_manager.out_log:
            print('=' * 30 + '=>\tLoading Checkpoint {}'.format(checkpoint_file))
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_file)
        else:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')

        model_dict = self.net.state_dict()
        model_dict.update(checkpoint['state_dict'])
        self.net.load_state_dict(model_dict)

        # TODO:  why set new manual seed
        new_manual_seed = int(time.time())
        set_manual_seed(new_manual_seed)

        self.start_epoch = checkpoint['start_epochs']
        self.monitor_metric, self.best_monitor = checkpoint['best_monitor']
        self.run_manager.optimizer.load_state_dict(checkpoint['weight_optimizer'])
        scheduler_dict = self.run_manager.scheduler.state_dict()
        scheduler_dict.update(checkpoint['weight_scheduler'])
        self.run_manager.scheduler.load_state_dict(scheduler_dict)
        self.arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
        self.warm_up = checkpoint['warmup']
        if self.run_manager.out_log:
            print('=' * 30 + '=>\tLoaded Checkpoint {}'.format(checkpoint_file))

    def save_model(self, epoch, is_warmup, is_best, checkpoint_file_name):
        # TODO: wheter has arch_scheduler or not
        checkpoint = {
            'arch_optimizer': self.arch_optimizer.state_dict(),
            #'arch_scheduler': self.arch_scheduler.state_dict(), # does not have arch_scheduler
        }
        # saved in /train_search/ckpt_file
        self.run_manager.save_model(epoch, checkpoint, is_warmup=is_warmup, is_best=is_best, checkpoint_file_name=checkpoint_file_name)

    ''' training related methods '''
    def validate(self):
        # TODO: use validate method in run_manager, after perform model derivation.
        # for validation phrase, not train search phrase, only performing validation
        # valid_loader batch_size = test_batch_size
        # have already equals to test_batch_size in DataProvider
        self.run_manager.run_config.valid_loader.batch_sampler.batch_size = self.run_manager.run_config.train_batch_size
        self.run_manager.run_config.valid_loader.batch_sampler.drop_last = False

        # TODO: network_level and cell_level decode in net_flops() method
        # TODO: valid on validation set (from training set) under train mode
        # only have effect on operation related to train mode
        # like bn or dropout
        loss, acc, miou, fscore = self.run_manager.validate(is_test=False, net=self.net, use_train_mode=True)

        # TODO: network flops and network param count should be calculated after the best network derived.
        # flops = self.run_manager.net_flops()
        # params = count_parameters(self.net)
        # target_hardware is None by default
        if self.arch_search_config.target_hardware in ['flops', None]:
            latency = 0
        else:
            raise NotImplementedError

        return loss, acc, miou, fscore, #flops, params

    def warm_up(self, warmup_epochs):
        if warmup_epochs <=0 :
            self.logger.log('=> warmup close', mode='warm')
            #print('\twarmup close')
            return
        # set optimizer and scheduler in warm_up phase
        lr_max = self.arch_search_config.warmup_lr
        data_loader = self.run_manager.run_config.train_loader
        scheduler_params = self.run_manager.run_config.optimizer_config['scheduler_params']
        optimizer_params = self.run_manager.run_config.optimizer_config['optimizer_params']
        momentum, nesterov, weight_decay = optimizer_params['momentum'], optimizer_params['nesterov'], optimizer_params['weight_decay']
        eta_min = scheduler_params['eta_min']
        optimizer_warmup = torch.optim.SGD(self.net.weight_parameters(), lr_max, momentum, weight_decay=weight_decay, nesterov=nesterov)
        # set initial_learning_rate in weight_optimizer
        #for param_groups in self.run_manager.optimizer.param_groups:
        #    param_groups['lr'] = lr_max
        lr_scheduler_warmup = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_warmup, warmup_epochs, eta_min)
        iter_per_epoch = len(data_loader)
        total_iteration = warmup_epochs * iter_per_epoch

        self.logger.log('=> warmup begin', mode='warm')

        epoch_time = AverageMeter()
        end_epoch = time.time()
        for epoch in range(self.warmup_epoch, warmup_epochs):
            self.logger.log('\n'+'-'*30+'Warmup Epoch: {}'.format(epoch+1)+'-'*30+'\n', mode='warm')

            lr_scheduler_warmup.step(epoch)
            warmup_lr = lr_scheduler_warmup.get_lr()
            self.net.train()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accs = AverageMeter()
            mious = AverageMeter()
            fscores = AverageMeter()

            epoch_str = 'epoch[{:03d}/{:03d}]'.format(epoch + 1, warmup_epochs)
            time_left = epoch_time.average * (warmup_epochs - epoch)
            common_log = '[Warmup the {:}] Left={:} LR={:}'.format(epoch_str, str(timedelta(seconds=time_left)) if epoch!=0 else None, warmup_lr)
            self.logger.log(common_log, mode='warm')
            end = time.time()

            # single_path init
            _, network_index = self.net.get_network_arch_hardwts_with_constraint()
            _, aspp_index = self.net.get_aspp_hardwts_index()
            single_path = self.net.sample_single_path(self.run_manager.run_config.nb_layers, aspp_index, network_index)

            for i, (datas, targets) in enumerate(data_loader):
                #print(i)
                #print(self.net.single_path)
                #if i == 59: # used for debug
                #    break
                if torch.cuda.is_available():
                    datas = datas.to(self.run_manager.device, non_blocking=True)
                    targets = targets.to(self.run_manager.device, non_blocking=True)
                else:
                    raise ValueError('do not support cpu version')
                data_time.update(time.time()-end)

                # TODO: update one architecture sufficiently
                # 1. get hardwts and index
                # 2. sample single_path, and set single_path
                # 3. get arch_sample_frequency
                # 4. update single_path per '{:}'.format(sample_arch_frequency) frequency
                if (i+1) % self.arch_search_config.sample_arch_frequency == 0:
                    _, network_index = self.net.get_network_arch_hardwts_with_constraint()
                    _, aspp_index = self.net.get_aspp_hardwts_index()
                    single_path = self.net.sample_single_path(self.run_manager.run_config.nb_layers, aspp_index, network_index)

                logits = self.net.single_path_forward(datas, single_path)

                # TODO: don't add entropy reg in warmup_phase

                ce_loss = self.run_manager.criterion(logits, targets)
                #entropy_reg = self.net.calculate_entropy(single_path)
                #cell_entropy, network_entropy, _ = self.net.calculate_entropy(single_path)
                loss = self.run_manager.add_regularization_loss(ce_loss, None)
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

                self.net.zero_grad()
                loss.backward()
                self.run_manager.optimizer.step()

                batch_time.update(time.time()-end)
                end = time.time()
                if (i+1) % self.run_manager.run_config.train_print_freq == 0 or i + 1 == iter_per_epoch:
                    Wstr = '|*WARM-UP*|' + time_string() + '[{:}][iter{:03d}/{:03d}]'.format(epoch_str, i+1, iter_per_epoch)
                    Tstr = '|Time     | [{batch_time.val:.2f} ({batch_time.avg:.2f})  Data {data_time.val:.2f} ({data_time.avg:.2f})]'.format(batch_time=batch_time, data_time=data_time)
                    Bstr = '|Base     | [Loss {loss.val:.3f} ({loss.avg:.3f})  Accuracy {acc.val:.2f} ({acc.avg:.2f}) MIoU {miou.val:.2f} ({miou.avg:.2f}) F {fscore.val:.2f} ({fscore.avg:.2f})]'\
                        .format(loss=losses, acc=accs, miou=mious, fscore=fscores)
                    self.logger.log(Wstr+'\n'+Tstr+'\n'+Bstr, 'warm')

            #torch.cuda.empty_cache()
            epoch_time.update(time.time() - end_epoch)
            end_epoch = time.time()

            #epoch_str = '{:03d}/{:03d}'.format(epoch+1, self.run_manager.run_config.warmup_epochs)
            log = '[{:}] warm :: loss={:.2f} accuracy={:.2f} miou={:.2f} f1score={:.2f}\n'.format(
                epoch_str, losses.average, accs.average, mious.average, fscores.average)
            self.vis.visdom_update(epoch, 'warmup_loss', [losses.average])
            self.vis.visdom_update(epoch, 'warmup_miou', [mious.average])

            self.logger.log(log, mode='warm')

            '''
            # TODO: wheter perform validation after each epoch in warmup phase ?
            valid_loss, valid_acc, valid_miou, valid_fscore = self.validate()
            valid_log = 'Warmup Valid\t[{0}/{1}]\tLoss\t{2:.6f}\tAcc\t{3:6.4f}\tMIoU\t{4:6.4f}\tF\t{5:6.4f}'\
                .format(epoch+1, warmup_epochs, valid_loss, valid_acc, valid_miou, valid_fscore)
                        #'\tflops\t{6:}M\tparams{7:}M'\
            valid_log += 'Train\t[{0}/{1}]\tLoss\t{2:.6f}\tAcc\t{3:6.4f}\tMIoU\t{4:6.4f}\tFscore\t{5:6.4f}'
            self.run_manager.write_log(valid_log, 'valid')
            '''


            # continue warmup phrase
            self.warmup = epoch + 1 < warmup_epochs
            self.warmup_epoch = self.warmup_epoch + 1
            #self.start_epoch = self.warmup_epoch
            # To save checkpoint in warmup phase at specific frequency.
            if (epoch+1) % self.run_manager.run_config.save_ckpt_freq == 0 or (epoch+1) == warmup_epochs:
                state_dict = self.net.state_dict()
                # rm architecture parameters because, in warm_up phase, arch_parameters are not updated.
                #for key in list(state_dict.keys()):
                #    if 'cell_arch_parameters' in key or 'network_arch_parameters' in key or 'aspp_arch_parameters' in key:
                #        state_dict.pop(key)
                checkpoint = {
                    'state_dict': state_dict,
                    'weight_optimizer' : self.run_manager.optimizer.state_dict(),
                    'weight_scheduler': self.run_manager.optimizer.state_dict(),
                    'warmup': self.warmup,
                    'warmup_epoch': epoch+1,
                }
                filename = self.logger.path(mode='warm', is_best=False)
                save_path = save_checkpoint(checkpoint, filename, self.logger, mode='warm')
                # TODO: save_path used to resume last info

    def train(self, fix_net_weights=False):

        # have config valid_batch_size, and ignored drop_last.
        data_loader = self.run_manager.run_config.train_loader
        iter_per_epoch = len(data_loader)
        total_iteration = iter_per_epoch * self.run_manager.run_config.epochs
        self.update_scheduler = self.arch_search_config.get_update_schedule(iter_per_epoch)

        if fix_net_weights: # used to debug
            data_loader = [(0, 0)] * iter_per_epoch
            print('Train Phase close for debug')

        # arch_parameter update frequency and times in each iteration.
        #update_schedule = self.arch_search_config.get_update_schedule(iter_per_epoch)

        # pay attention here, total_epochs include warmup epochs
        epoch_time = AverageMeter()
        end_epoch = time.time()
        # TODO : use start_epochs

        # single_path init
        _, network_index = self.net.get_network_arch_hardwts_with_constraint()
        _, aspp_index = self.net.get_aspp_hardwts_index()
        single_path = self.net.sample_single_path(self.run_manager.run_config.nb_layers, aspp_index, network_index)

        for epoch in range(self.start_epoch, self.run_manager.run_config.epochs):
            self.logger.log('\n'+'-'*30+'Train Epoch: {}'.format(epoch+1)+'-'*30+'\n', mode='search')
            self.run_manager.scheduler.step(epoch)
            train_lr = self.run_manager.scheduler.get_lr()
            arch_lr = self.arch_optimizer.param_groups[0]['lr']
            self.net.set_tau(self.arch_search_config.tau_max - (self.arch_search_config.tau_max - self.arch_search_config.tau_min) * (epoch) / (self.run_manager.run_config.epochs))
            tau = self.net.get_tau()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            accs = AverageMeter()
            mious = AverageMeter()
            fscores = AverageMeter()

            #valid_data_time = AverageMeter()
            valid_losses = AverageMeter()
            valid_accs = AverageMeter()
            valid_mious = AverageMeter()
            valid_fscores = AverageMeter()

            self.net.train()

            epoch_str = 'epoch[{:03d}/{:03d}]'.format(epoch + 1, self.run_manager.run_config.epochs)
            time_left = epoch_time.average * (self.run_manager.run_config.epochs - epoch)
            common_log = '[*Train-Search* the {:}] Left={:} WLR={:} ALR={:} tau={:}'\
                .format(epoch_str, str(timedelta(seconds=time_left)) if epoch != 0 else None, train_lr, arch_lr, tau)
            self.logger.log(common_log, 'search')

            end = time.time()

            for i, (datas, targets) in enumerate(data_loader):
                #print(self.net.single_path)
                #print(i)
                #if i == 59: break
                if not fix_net_weights:
                    if torch.cuda.is_available():
                        datas = datas.to(self.run_manager.device, non_blocking=True)
                        targets = targets.to(self.run_manager.device, non_blocking=True)
                    else:
                        raise ValueError('do not support cpu version')
                    data_time.update(time.time() - end)
                    '''
                    if (i + 1) % self.arch_search_config.sample_arch_frequency == 0:
                        _, network_index = self.net.get_network_arch_hardwts_with_constraint()
                        _, aspp_index = self.net.get_aspp_hardwts_index()
                        single_path = self.net.sample_single_path(self.run_manager.run_config.nb_layers, aspp_index, network_index)
                    '''
                    logits = self.net.single_path_forward(datas, single_path) # super network gdas forward
                    # loss
                    ce_loss = self.run_manager.criterion(logits, targets)
                    #cell_reg, network_reg, _ = self.net.calculate_entropy(single_path) # todo: pay attention, entropy is unnormalized, should use small lambda
                    #print('entropy_reg:', entropy_reg)
                    loss = self.run_manager.add_regularization_loss(ce_loss, None)
                    #loss = self.run_manager.criterion(logits, targets)
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

                    self.net.zero_grad()
                    loss.backward()

                    self.run_manager.optimizer.step()

                    if (i+1) % self.arch_search_config.sample_arch_frequency == 0 or (i+1) == iter_per_epoch: # at the i-th iteration, update arch_parameters update_scheduler[i] times.
                        valid_datas, valid_targets = self.run_manager.run_config.valid_next_batch
                        if torch.cuda.is_available():
                            valid_datas = valid_datas.to(self.run_manager.device, non_blocking=True)
                            valid_targets = valid_targets.to(self.run_manager.device, non_blocking=True)
                        else:
                            raise ValueError('do not support cpu version')

                        _, network_index = self.net.get_network_arch_hardwts_with_constraint() # set self.hardwts again
                        _, aspp_index = self.net.get_aspp_hardwts_index()
                        single_path = self.net.sample_single_path(self.run_manager.run_config.nb_layers, aspp_index, network_index)
                        logits = self.net.single_path_forward(valid_datas, single_path)

                        ce_loss = self.run_manager.criterion(logits, valid_targets)
                        cell_reg, network_reg, _ = self.net.calculate_entropy(single_path)
                        loss = self.run_manager.add_regularization_loss(ce_loss, [cell_reg, network_reg])

                        # metrics and update
                        valid_evaluator = Evaluator(self.run_manager.run_config.nb_classes)
                        valid_evaluator.add_batch(valid_targets, logits)
                        acc = valid_evaluator.Pixel_Accuracy()
                        miou = valid_evaluator.Mean_Intersection_over_Union()
                        fscore = valid_evaluator.Fx_Score()
                        valid_losses.update(loss.data.item(), datas.size(0))
                        valid_accs.update(acc.item(), datas.size(0))
                        valid_mious.update(miou.item(), datas.size(0))
                        valid_fscores.update(fscore.item(), datas.size(0))

                        self.net.zero_grad()
                        loss.backward() # release computational graph
                        # update arch_parameters per '{:}'.format(arch_param_update_frequency)
                        self.arch_optimizer.step()

                    # batch_time of one iter of train and valid.
                    batch_time.update(time.time() - end)
                    end = time.time()

                    # in other case, calculate metrics normally
                    # train_print_freq == sample_arch_freq
                    if (i+1) % self.run_manager.run_config.train_print_freq == 0 or (i + 1) == iter_per_epoch:
                        Wstr = '|*Search*|' + time_string() + '[{:}][iter{:03d}/{:03d}]'.format(epoch_str, i + 1, iter_per_epoch)
                        Tstr = '|Time    | {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
                        Bstr = '|Base    | [Loss {loss.val:.3f} ({loss.avg:.3f}) Accuracy {acc.val:.2f} ({acc.avg:.2f}) MIoU {miou.val:.2f} ({miou.avg:.2f}) F {fscore.val:.2f} ({fscore.avg:.2f})]'.format(loss=losses, acc=accs, miou=mious, fscore=fscores)
                        Astr = '|Arch    | [Loss {loss.val:.3f} ({loss.avg:.3f}) Accuracy {acc.val:.2f} ({acc.avg:.2f}) MIoU {miou.val:.2f} ({miou.avg:.2f}) F {fscore.val:.2f} ({fscore.avg:.2f})]'.format(loss=valid_losses, acc=valid_accs, miou=valid_mious, fscore=valid_fscores)
                        self.logger.log(Wstr+'\n'+Tstr+'\n'+Bstr+'\n'+Astr, mode='search')

            _, network_index = self.net.get_network_arch_hardwts_with_constraint()  # set self.hardwts again
            _, aspp_index = self.net.get_aspp_hardwts_index()
            single_path = self.net.sample_single_path(self.run_manager.run_config.nb_layers, aspp_index, network_index)
            cell_arch_entropy, network_arch_entropy, total_entropy = self.net.calculate_entropy(single_path)

            # update visdom
            if self.vis is not None:
                self.vis.visdom_update(epoch, 'loss', [losses.average, valid_losses.average])
                self.vis.visdom_update(epoch, 'accuracy', [accs.average, valid_accs.average])
                self.vis.visdom_update(epoch, 'miou', [mious.average, valid_mious.average])
                self.vis.visdom_update(epoch, 'f1score', [fscores.average, valid_fscores.average])

                self.vis.visdom_update(epoch, 'cell_entropy', [cell_arch_entropy])
                self.vis.visdom_update(epoch, 'network_entropy', [network_arch_entropy])
                self.vis.visdom_update(epoch, 'entropy', [total_entropy])

            #torch.cuda.empty_cache()
            # update epoch_time
            epoch_time.update(time.time()-end_epoch)
            end_epoch = time.time()

            epoch_str = '{:03d}/{:03d}'.format(epoch+1, self.run_manager.run_config.epochs)
            log = '[{:}] train :: loss={:.2f} accuracy={:.2f} miou={:.2f} f1score={:.2f}\n' \
                  '[{:}] valid :: loss={:.2f} accuracy={:.2f} miou={:.2f} f1score={:.2f}\n'.format(
                epoch_str, losses.average, accs.average, mious.average, fscores.average,
                epoch_str, valid_losses.average, valid_accs.average, valid_mious.average, valid_fscores.average
            )
            self.logger.log(log, mode='search')

            self.logger.log('<<<---------->>> Super Network decoding <<<---------->>> ', mode='search')
            actual_path, cell_genotypes = self.net.network_cell_arch_decode()
            #print(cell_genotypes)
            new_genotypes = []
            for _index, genotype in cell_genotypes:
                xlist = []
                print(_index, genotype)
                for edge_genotype in genotype:
                    for (node_str, select_index) in edge_genotype:
                        xlist.append((node_str, self.run_manager.run_config.conv_candidates[select_index]))
                new_genotypes.append((_index, xlist))
            log_str = 'The {:} decode network:\n' \
                      'actual_path = {:}\n' \
                      'genotype:'.format(epoch_str, actual_path)
            for _index, genotype in new_genotypes:
                log_str += 'index: {:} arch: {:}\n'.format(_index, genotype)
            self.logger.log(log_str, mode='network_space', display=False)

            # TODO： perform save the best network ckpt
            # 1. save network_arch_parameters and cell_arch_parameters
            # 2. save weight_parameters
            # 3. weight_optimizer.state_dict
            # 4. arch_optimizer.state_dict
            # 5. training process
            # 6. monitor_metric and the best_value
            # get best_monitor in valid phase.
            val_monitor_metric = get_monitor_metric(self.run_manager.monitor_metric, valid_losses.average,
                                                    valid_accs.average,
                                                    valid_mious.average, valid_fscores.average)
            is_best = self.run_manager.best_monitor < val_monitor_metric
            self.run_manager.best_monitor = max(self.run_manager.best_monitor, val_monitor_metric)
            # 1. if is_best : save_current_ckpt
            # 2. if can be divided : save_current_ckpt

            #self.run_manager.save_model(epoch, {
            #    'arch_optimizer': self.arch_optimizer.state_dict(),
            #}, is_best=True, checkpoint_file_name=None)
            # TODO: have modification on checkpoint_save semantics
            if (epoch + 1) % self.run_manager.run_config.save_ckpt_freq == 0 or (epoch + 1) == self.run_manager.run_config.epochs or is_best:
                checkpoint = {
                    'state_dict'      : self.net.state_dict(),
                    'weight_optimizer': self.run_manager.optimizer.state_dict(),
                    'weight_scheduler': self.run_manager.scheduler.state_dict(),
                    'arch_optimizer'  : self.arch_optimizer.state_dict(),
                    'best_monitor'    : (self.run_manager.monitor_metric, self.run_manager.best_monitor),
                    'warmup'          : False,
                    'start_epochs'    : epoch + 1,
                }
                checkpoint_arch = {
                    'actual_path'   : actual_path,
                    'cell_genotypes': cell_genotypes,
                }
                filename = self.logger.path(mode='search', is_best=is_best)
                filename_arch = self.logger.path(mode='arch', is_best=is_best)
                save_checkpoint(checkpoint, filename, self.logger, mode='search')
                save_checkpoint(checkpoint_arch, filename_arch, self.logger, mode='arch')
