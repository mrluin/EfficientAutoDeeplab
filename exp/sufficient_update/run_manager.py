import time
import os
import math
import torch
import logging
import torch.nn as nn
import numpy as np
import json
import torch.optim as optim
from datetime import timedelta
from data.WHUBuilding import WHUBuildingDataProvider
from utils.common import set_manual_seed
from utils.common import get_monitor_metric
from utils.common import AverageMeter, convert_secs2time
from utils.common import count_parameters
from utils.logger import time_string, save_checkpoint
from utils.metrics import Evaluator
from utils.calculators import calculate_weights_labels
from optimizers import CosineAnnealingLR, MultiStepLR, LinearLR, CrossEntropyLabelSmooth, ExponentialLR
from models.gumbel_cells import autodeeplab, proxyless, counter, my_search_space
from PIL import Image
'''
# RunConfig: 1. all the configurations from args
#            2. build optimizer, learning_rate, and dataset
#
# RunManager: 1. manage to train, valid and test
#             2. processes related to training phrase
'''
class RunConfig:
    def __init__(self, path, epochs, warmup_epochs, gpu_ids, workers,
                 save_path, dataset, nb_classes,
                 train_batch_size, valid_batch_size, test_batch_size, valid_size,
                 ori_size, crop_size,
                 #init_lr, #scheduler, #scheduler_params,
                 optimizer_config,
                 model_init, init_div_groups, filter_multiplier, block_multiplier, steps, bn_momentum, bn_eps, dropout, nb_layers,
                 validation_freq, train_print_freq, save_ckpt_freq, monitor,
                 #print_arch_param_step_freq,
                 use_unbalanced_weights,
                 search_space,
                 #conv_candidates,
                 reg_loss_type, reg_loss_params,
                 actual_path = None, cell_genotypes = None,
                 search_resume = False, retrain_resume = False, evaluation = False,
                 **kwargs):

        # actual_path and cell_genotypes are used in retrain-phase

        self.path = path

        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        #self.total_epochs = self.epochs + self.warmup_epochs
        self.gpu_ids = gpu_ids
        self.workers = workers

        self.save_path = save_path
        self.dataset = dataset
        self.nb_classes = nb_classes
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_batch_size = test_batch_size
        self.valid_size = valid_size

        self.ori_size = ori_size
        self.crop_size = crop_size

        #self.init_lr = init_lr
        #self.lr_scheduler = scheduler
        #self.lr_scheduler_param = scheduler_params

        self.optimizer_config = optimizer_config


        #self.no_decay_keys = no_decay_keys

        self.model_init = model_init
        self.init_div_groups = init_div_groups
        self.filter_multiplier = filter_multiplier
        self.block_multiplier = block_multiplier
        self.steps = steps
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.dropout = dropout
        self.nb_layers = nb_layers

        self.validation_freq = validation_freq
        self.train_print_freq = train_print_freq
        self.save_ckpt_freq = save_ckpt_freq
        self.monitor = monitor
        #self.print_arch_param_step_freq = print_arch_param_step_freq

        self.use_unbalanced_weights = use_unbalanced_weights

        self.search_space = search_space
        if self.search_space == 'autodeeplab':
            self.conv_candidates = autodeeplab
        elif self.search_space == 'proxyless':
            self.conv_candidates = proxyless
        elif self.search_space == 'counter':
            self.conv_candidates = counter
        elif self.search_space == 'my_search_space':
            self.conv_candidates = my_search_space
        else:
            raise ValueError('search space {:} is not support'.format(self.search_space))
        #self.conv_candidates = search_space_dict[self.search_space]

        self.reg_loss_type = reg_loss_type
        self.reg_loss_params = reg_loss_params

        self.actual_path = actual_path
        self.cell_genotypes = cell_genotypes

        self.search_resume = search_resume
        self.retrain_resume = retrain_resume
        self.evaluation = evaluation



        self._data_provider = None
        self._train_iter, self._valid_iter, self._test_iter = None, None, None
    @property
    def config(self):
        config = {
            #'type': type(self)
        }
        for key in self.__dict__:
            # SGD cannot be serializable, so optimizer is excluded
            # if not key.startswith('_') and not key.startswith('optimizer'):
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    ''' get weight_optimizer, scheduler, and training criterion '''
    def get_optim_scheduler_criterion(self, parameters, optimizer_config):

        # only for weight, optimizer config needs:
        # 1. weight_optimizer_type
        # 2. weight_init_lr
        # 3. optimizer params
        # 4. weight_scheduler
        # 5. scheduler params
        # 6. criterion type
        # 7. criterion params

        # todo if it has no_decay_keys
        if optimizer_config['optimizer_type'] == 'SGD':
            optimizer_params = optimizer_config['optimizer_params']
            momentum, nesterov, weight_decay = optimizer_params.get('momentum'), optimizer_params.get('nesterov'), optimizer_params.get('weight_decay')
            optimizer = torch.optim.SGD(parameters, optimizer_config['init_lr'], momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        elif optimizer_config['optimizer_type'] == 'RMSprop':
            optimizer_params = optimizer_config['optimizer_params']
            momentum, weight_decay = optimizer_params.get('momentum'), optimizer_params.get('weight_decay')
            optimizer = torch.optim.RMSprop(parameters, optimizer_config['init_lr'], momentum=momentum, weight_decay=weight_decay)
        elif optimizer_config['optimizer_type'] == 'Adam':
            optimizer = torch.optim.Adam(parameters, optimizer_config['init_lr'])
            # has issue in Adam optimizer
            #optimizer = torch.optim.Adam(parameters, optimizer_config.init_lr, **optimizer_config.optimizer_params)
        else:
            raise ValueError('invalid optim : {:}'.format(optimizer_config.optimizer_type))

        if optimizer_config['scheduler'] == 'cosine':
            scheduler_params = optimizer_config['scheduler_params']
            T_max = scheduler_params['T_max'] if scheduler_params.get('T_max') is not None else optimizer_config.get('epochs')
            eta_min = scheduler_params['eta_min']
            #scheduler = CosineAnnealingLR(optimizer, optimizer_config.warmup, optimizer_config.epochs, T_max, scheduler_params.eta_min)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)
        elif optimizer_config['scheduler'] == 'multistep':
            scheduler_params = optimizer_config['scheduler_params']
            milestones, gammas = scheduler_params['milestones'], scheduler_params['gammas']
            #scheduler = MultiStepLR(optimizer, optimizer_config.warmup, optimizer_config.epochs, scheduler_params.milestones, scheduler_params.gammas)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gammas)
        elif optimizer_config['scheduler'] == 'exponential':
            scheduler_params = optimizer_config['scheduler_params']
            gamma = scheduler_params['gamma']
            #scheduler = ExponentialLR(optimizer, optimizer_config.warmup, optimizer_config.epochs, scheduler_params.gamma)
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
        elif optimizer_config['scheduler'] == 'linear':
            raise NotImplementedError
        elif optimizer_config['scheduler'] == 'poly':
            lambda1 = lambda epoch: pow((1 - epoch / optimizer_config['epochs']), 0.9)
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
        else:
            raise ValueError('invalid scheduler : {:}'.format(optimizer_config.scheduler))

        if optimizer_config['criterion'] == 'Softmax':
            criterion = torch.nn.CrossEntropyLoss().to('cuda:{}'.format(self.gpu_ids))
        elif optimizer_config['criterion'] == 'SmoothSoftmax':
            criterion_params = optimizer_config['criterion_params']
            criterion = CrossEntropyLabelSmooth(optimizer_config.class_num, criterion_params.label_smooth).to('cuda:{}'.format(self.gpu_ids))
        elif optimizer_config['criterion'] == 'WeightedSoftmax':
            classes_weights = calculate_weights_labels(self.path, 'WHUBuilding', self.train_loader, self.nb_classes)
            criterion = torch.nn.CrossEntropyLoss(weight=classes_weights).to('cuda:{}'.format(self.gpu_ids))
        else:
            raise ValueError('invalid criterion : {:}'.format(optimizer_config.criterion))

        return optimizer, scheduler, criterion

    ''' data provider '''
    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_batch_size': self.valid_batch_size,
            'valid_size': self.valid_size,
            'nb_works': self.workers,
            'save_path': self.save_path,
        }
    @property
    def data_provider(self):
        if self._data_provider is None:
            if self.dataset == 'WHUBuilding':
                self._data_provider = WHUBuildingDataProvider(**self.data_config)
            else:
                raise ValueError('do not support: {}'.format(self.dataset))
        return self._data_provider

    @data_provider.setter
    def data_provider(self, val):
        self.data_provider = val

    @property
    def train_loader(self):
        return self.data_provider.train_loader

    @property
    def valid_loader(self):
        return self.data_provider.valid_loader

    @property
    def test_loader(self):
        return self.data_provider.test_loader

    @property
    def train_next_batch(self):
        if self._train_iter is None:
            self._train_iter = iter(self.train_loader)
        try:
            data = next(self._train_iter)
            # ending of data_loader & reset
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            data = next(self._train_iter)
        return data

    @property
    def valid_next_batch(self):
        if self._valid_iter is None:
            self._valid_iter = iter(self.valid_loader)
        try:
            data = next(self._valid_iter)
            # ending of data_loader & reset
        except StopIteration:
            self._valid_iter = iter(self.valid_loader)
            data = next(self._valid_iter)
        return data

    @property
    def test_next_batch(self):
        if self._test_iter is None:
            self._test_iter = iter(self.test_loader)
        try:
            data = next(self._test_iter)
            # ending of data_loader & reset
        except StopIteration:
            self._test_iter = iter(self.test_loader)
            data = next(self._test_iter)
        return data



class RunManager:
    def __init__(self, path, super_network, logger, run_config: RunConfig, vis=None, out_log=True):

        self.path = path # root path to workspace

        self.model_path = logger.model_dir
        self.log_path = logger.log_dir
        self.prediction_path = logger.predictions_dir

        #os.makedirs(self.ckpt_save_path, exist_ok=True)
        #os.makedirs(self.log_save_path, exist_ok=True)
        #os.makedirs(self.prediction_save_path, exist_ok=True)

        self.model = super_network
        self.logger = logger
        self.run_config = run_config
        self.out_log = out_log
        self.vis = vis


        self.best_monitor = self.build_monitor(self.run_config.monitor)
        self.start_epoch = 0

        # initialize model
        # 1. add test flag
        # 2. don't perform initialization in resume case (search, retrain), and in evaluation case.
        if self.run_config.search_resume == False and self.run_config.retrain_resume == False and self.run_config.evaluation == False:
            self.model.init_model(self.run_config.model_init, self.run_config.init_div_groups)
            self.logger.log('=> SuperNetwork operation weight initialization ... ... ', mode='info')
        else:
            self.logger.log('=> SuperNetwork operation weight has resume from checkpoint file, skip initialization ... ... ', mode='info')


        if torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(self.run_config.gpu_ids))
            print('Device: {}'.format(self.device))
            self.model.to(self.device)
        else:
            raise ValueError('do not support cpu version')

        # get optimizer, scheduler, and loss function
        optimizer, scheduler, criterion = self.run_config.get_optim_scheduler_criterion(
            parameters=self.model.weight_parameters(), optimizer_config=self.run_config.optimizer_config
        )

        # only used for training counter-network
        #self.optimizer = torch.optim.Adam(self.model.weight_parameters(), lr=5e-4, weight_decay=2e-4)
        #lambda1 = lambda epoch: pow((1-((epoch-1)/self.run_config.epochs)), 0.9)
        #self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    ''' save path and log path '''
    @property
    def get_model_path(self):
        return self.model_path

    @property
    def get_log_path(self):
        return self.log_path

    @property
    def get_prediction_path(self):
        return self.prediction_path

    def build_monitor(self, monitor):
        monitor_mode = monitor.split('#')[0]
        self.monitor_metric = monitor.split('#')[1]
        assert self.monitor_metric in ['miou', 'fscore'], 'invalid monitor metric'
        best_monitor = math.inf if monitor_mode == 'min' else -math.inf
        return best_monitor


    def add_regularization_loss(self, epoch, ce_loss, reg_value=None):
        # TODO: add entropy_reg
        # 1. lambda for both cell_entropy and network_entropy
        # 2. lambda1 for cell, and lambda2 for network separately.

        if reg_value is None:
            return ce_loss

        if self.run_config.reg_loss_type == 'add#linear':
            reg_lambda1, reg_lambda2 = self.run_config.reg_loss_params['lambda1'], self.run_config.reg_loss_params['lambda2']
            reg_loss = reg_lambda1 * reg_value[0] + reg_lambda2 * reg_value[1]
            return ce_loss + reg_loss
        elif self.run_config.reg_loss_type == 'add#linear#linearschedule':
            # TODO: add lambda scheduler, reg_lambda linearly increase to defined value from zero in epochs.
            reg_lambda1, reg_lambda2 = self.run_config.reg_loss_params['lambda1'], self.run_config.reg_loss_params['lambda2']
            lambda1 = reg_lambda1 * epoch / (self.run_config.epochs - 1)
            lambda2 = reg_lambda2 * epoch / (self.run_config.epochs - 1)
            reg_loss = lambda1 * reg_value[0] + lambda2 * reg_value[1]
            return ce_loss + reg_loss
        elif self.run_config.reg_loss_type == 'mul#log':
            raise NotImplementedError
            #alpah = self.run_config.reg_loss_params['alpha']
            #beta = self.run_config.reg_loss_params['beta']
        elif self.run_config.reg_loss_type == 'none':
            return ce_loss
        else:
            raise ValueError('reg_loss_type: {:} is not supported'.format(self.run_config.reg_loss_type))


    ''' net info '''
    def net_flops(self):
        # TODO: get rid of, have using hook function to replace.
        # TODO: get flops of specific architecture, related to architecture parameter

        data_shape = [1] + list(self.run_config.data_provider.data_shape)
        # data_shape [1, 3, 512, 512]

        net = self.net
        #
        input_var = torch.zeros(data_shape, device=self.device)
        with torch.no_grad():
            flop, _ = net.get_flops(input_var)
        return flop

    def load_model(self, checkpoint_file):
        # TODO: get rid of load_model. load resume_file have completed in the main function in each phase.
        # only used in run_manager
        assert checkpoint_file is not None and os.path.exists(checkpoint_file),\
            'checkpoint_file can not be found'
        print('=' * 30 + '=>\tLoading Checkpoint {}'.format(checkpoint_file))
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_file)
        else:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
        model_dict = self.model.state_dict()
        model_dict.update(checkpoint['state_dict'])
        self.model.load_state_dict(model_dict)

        # TODO: why set new manual seed
        new_manual_seed = int(time.time())
        set_manual_seed(new_manual_seed)

        self.start_epoch = checkpoint['start_epochs']
        self.monitor_metric, self.best_monitor = checkpoint['best_monitor']
        self.optimizer.load_state_dict(checkpoint['weight_optimizer'])
        scheduler_dict = self.scheduler.state_dict()
        scheduler_dict.update(checkpoint['weight_scheduler'])
        self.scheduler.load_state_dict(scheduler_dict)

        # TODO: something should loaded in nas_manager, related to warm_up, train search, and arch_optimizer info

        print('=' * 30 + '=>\tLoaded Checkpoint {}'.format(checkpoint_file))

    def _save_pred(self, predictions, filenames):

        for index, map in enumerate(predictions):
            map = torch.argmax(map, dim=0)
            map = map * 255
            map = np.asarray(map.cpu(), dtype=np.uint8)
            map = Image.fromarray(map)
            # filename /0.1.png [0] 0 [1] 1
            filename = filenames[index].split('/')[-1].split('.')
            save_filename = filename[0] + '.' + filename[1]
            save_path = os.path.join(self.prediction_path, 'patches', save_filename + '.png')
            map.save(save_path)

    def validate(self, epoch=None, is_test=False, use_train_mode=False):
        # 1. super network viterbi_decodde, get actual_path
        # 2. cells genotype decode, which are on the actual_path in the super network
        # 3. according to actual_path and cells genotypes, construct the best network.
        # 4. use the best network, to perform test phrase.

        if is_test:
            data_loader = self.run_config.test_loader
            epoch_str = None
            self.logger.log('\n' + '-' * 30 + 'TESTING PHASE' + '-' * 30 + '\n', mode='valid')
        else:
            data_loader = self.run_config.valid_loader
            epoch_str = 'epoch[{:03d}/{:03d}]'.format(epoch + 1, self.run_config.epochs)
            self.logger.log('\n' + '-' * 30 + 'Valid epoch: {:}'.format(epoch_str) + '-' * 30 + '\n', mode='valid')

        model = self.model

        if use_train_mode: model.train()
        else: model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        mious = AverageMeter()
        fscores = AverageMeter()
        accs = AverageMeter()
        end0 = time.time()

        with torch.no_grad():
            if is_test:
                for i, (data, targets, filenames) in enumerate(data_loader):
                    if torch.cuda.is_available():
                        datas = datas.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)
                    else:
                        raise ValueError('do not support cpu version')
                    data_time.update(time.time()-end0)
                    logits = self.model(datas)
                    self._save_pred(logits, filenames)
                    loss = self.criterion(logits, targets)
                    # metrics calculate and update
                    evaluator = Evaluator(self.run_config.nb_classes)
                    evaluator.add_batch(targets, logits)
                    miou = evaluator.Mean_Intersection_over_Union()
                    fscore = evaluator.Fx_Score()
                    acc = evaluator.Pixel_Accuracy()
                    losses.update(loss.data.item(), datas.size(0))
                    mious.update(miou.item(), datas.size(0))
                    fscores.update(fscore.item(), datas.size(0))
                    accs.update(acc.item(), datas.size(0))
                    # duration
                    batch_time.update(time.time() - end0)
                    end0 = time.time()
                Wstr = '|*TEST*|' + time_string()
                Tstr = '|Time  | [{batch_time.val:.2f} ({batch_time.avg:.2f})  Data {data_time.val:.2f} ({data_time.avg:.2f})]'.format(batch_time=batch_time, data_time=data_time)
                Bstr = '|Base  | [Loss {loss.val:.3f} ({loss.avg:.3f})  Accuracy {acc.val:.2f} ({acc.avg:.2f}) MIoU {miou.val:.2f} ({miou.avg:.2f}) F {fscore.val:.2f} ({fscore.avg:.2f})]'.format(loss=losses, acc=accs, miou=mious, fscore=fscores)
                self.logger.log(Wstr + '\n' + Tstr + '\n' + Bstr, 'test')
            else:
                for i, (datas, targets) in enumerate(data_loader):
                    if torch.cuda.is_available():
                        datas = datas.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)
                    else:
                        raise ValueError('do not support cpu version')
                    data_time.update(time.time()-end0)
                    # validation of the derived model. normal forward pass.
                    logits = self.model(datas)

                    # TODO generate predictions
                    loss = self.criterion(logits, targets)

                    # metrics calculate and update
                    evaluator = Evaluator(self.run_config.nb_classes)
                    evaluator.add_batch(targets, logits)
                    miou = evaluator.Mean_Intersection_over_Union()
                    fscore = evaluator.Fx_Score()
                    acc = evaluator.Pixel_Accuracy()

                    losses.update(loss.data.item(), datas.size(0))
                    mious.update(miou.item(), datas.size(0))
                    fscores.update(fscore.item(), datas.size(0))
                    accs.update(acc.item(), datas.size(0))

                    # duration
                    batch_time.update(time.time() - end0)
                    end0 = time.time()
                Wstr = '|*VALID*|' + time_string() + epoch_str
                Tstr = '|Time   | [{batch_time.val:.2f} ({batch_time.avg:.2f})  Data {data_time.val:.2f} ({data_time.avg:.2f})]'.format(batch_time=batch_time, data_time=data_time)
                Bstr = '|Base   | [Loss {loss.val:.3f} ({loss.avg:.3f})  Accuracy {acc.val:.2f} ({acc.avg:.2f}) MIoU {miou.val:.2f} ({miou.avg:.2f}) F {fscore.val:.2f} ({fscore.avg:.2f})]'.format(loss=losses, acc=accs, miou=mious, fscore=fscores)
                self.logger.log(Wstr + '\n' + Tstr + '\n' + Bstr, 'valid')

        return losses.avg, accs.avg, mious.avg, fscores.avg

    def train_one_epoch(self, epoch):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        mious = AverageMeter()
        fscores = AverageMeter()

        epoch_str = 'epoch[{:03d}/{:03d}]'.format(epoch + 1, self.run_config.epochs)
        iter_per_epoch = len(self.run_config.train_loader)
        end = time.time()
        for i, (datas, targets) in enumerate(self.run_config.train_loader):
            if torch.cuda.is_available():
                datas = datas.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            else:
                raise ValueError('do not support cpu version')
            data_time.update(time.time()-end)
            logits = self.model(datas)
            loss = self.criterion(logits, targets)
            evaluator = Evaluator(self.run_config.nb_classes)
            evaluator.add_batch(targets, logits)
            acc = evaluator.Pixel_Accuracy()
            miou = evaluator.Mean_Intersection_over_Union()
            fscore = evaluator.Fx_Score()
            # metrics update
            losses.update(loss.data.item(), datas.size(0))
            accs.update(acc.item(), datas.size(0))
            mious.update(miou.item(), datas.size(0))
            fscores.update(fscore.item(), datas.size(0))
            # update parameters
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step() # only update network_weight_parameters
            # elapsed time
            batch_time.update(time.time()-end)
            end = time.time()
            # within one epoch, per iteration print train_log
            if (i+1) % self.run_config.train_print_freq == 0 or (i+1) == len(self.run_config.train_loader):
                #print(i+1, self.run_config.train_print_freq)
                Wstr = '|*TRAIN*|' + time_string() + '[{:}][iter{:03d}/{:03d}]'.format(epoch_str, i + 1, iter_per_epoch)
                Tstr = '|Time   | [{batch_time.val:.2f} ({batch_time.avg:.2f})  Data {data_time.val:.2f} ({data_time.avg:.2f})]'.format(batch_time=batch_time, data_time=data_time)
                Bstr = '|Base   | [Loss {loss.val:.3f} ({loss.avg:.3f})  Accuracy {acc.val:.2f} ({acc.avg:.2f}) MIoU {miou.val:.2f} ({miou.avg:.2f}) F {fscore.val:.2f} ({fscore.avg:.2f})]' \
                    .format(loss=losses, acc=accs, miou=mious, fscore=fscores)
                self.logger.log(Wstr+'\n'+Tstr+'\n'+Bstr, mode='retrain')
        return losses.avg, accs.avg, mious.avg, fscores.avg

    def train(self):
        epoch_time = AverageMeter()
        end = time.time()
        self.model.train()
        for epoch in range(self.start_epoch, self.run_config.epochs):
            self.logger.log('\n'+'-'*30+'Train epoch: {}'.format(epoch+1)+'-'*30+'\n', mode='retrain')
            epoch_str = 'epoch[{:03d}/{:03d}]'.format(epoch + 1, self.run_config.epochs)
            self.scheduler.step(epoch)
            train_lr = self.scheduler.get_lr()
            time_left = epoch_time.average * (self.run_config.epochs - epoch)
            common_log = '[Train the {:}] Left={:} LR={:}'.format(epoch_str, str(timedelta(seconds=time_left)) if epoch != 0 else None, train_lr)
            self.logger.log(common_log, 'retrain')
            # train log have been wrote in train_one_epoch
            loss, acc, miou, fscore = self.train_one_epoch(epoch)
            epoch_time.update(time.time()-end)
            end = time.time()
            # perform validation at the end of each epoch.
            val_loss, val_acc, val_miou, val_fscore = self.validate(epoch, is_test=False, use_train_mode=False)
            val_monitor_metric = get_monitor_metric(self.monitor_metric, val_loss, val_acc, val_miou, val_fscore)
            is_best = val_monitor_metric > self.best_monitor
            self.best_monitor = max(self.best_monitor, val_monitor_metric)
            # update visdom
            if self.vis is not None:
                self.vis.visdom_update(epoch, 'loss', [loss, val_loss])
                self.vis.visdom_update(epoch, 'accuracy', [acc, val_acc])
                self.vis.visdom_update(epoch, 'miou', [miou, val_miou])
                self.vis.visdom_update(epoch, 'f1score', [fscore, val_fscore])
            # save checkpoint
            if (epoch+1) % self.run_config.save_ckpt_freq == 0 or (epoch+1) == self.run_config.epochs or is_best:
                checkpoint = {
                    'state_dict'      : self.model.state_dict(),
                    'weight_optimizer': self.optimizer.state_dict(),
                    'weight_scheduler': self.scheduler.state_dict(),
                    'best_monitor'    : (self.monitor_metric, self.best_monitor),
                    'start_epoch'     : epoch+1,
                    'actual_path'     : self.run_config.actual_path,
                    'cell_genotypes'  : self.run_config.cell_genotypes
                }
                filename = self.logger.path(mode='retrain',is_best=is_best)
                save_checkpoint(checkpoint, filename, self.logger, mode='retrain')
