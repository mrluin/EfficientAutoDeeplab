import time
import os
import math
import torch
import logging
import torch.nn as nn
import json
import torch.optim as optim
from datetime import timedelta
from data.WHUBuilding import WHUBuildingDataProvider
from utils.common import set_manual_seed
from utils.common import get_monitor_metric
from utils.common import AverageMeter, convert_secs2time
from utils.common import count_parameters
from utils.metrics import Evaluator
from utils.calculators import calculate_weights_labels
from optimizers import CosineAnnealingLR, MultiStepLR, LinearLR, CrossEntropyLabelSmooth, ExponentialLR
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
                 conv_candidates,
                 **kwargs):

        self.path = path

        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.total_epochs = self.epochs + self.warmup_epochs
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

        self.conv_candidates = conv_candidates

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
            raise NotImplementedError
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
    def __init__(self, path, super_network, logger, run_config: RunConfig, out_log=True):

        self.path = path # root path to workspace
        self.model_path = logger.model_dir
        self.log_path = logger.log_dir
        self.prediction_path = logger.predictions_dir

        #os.makedirs(self.ckpt_save_path, exist_ok=True)
        #os.makedirs(self.log_save_path, exist_ok=True)
        #os.makedirs(self.prediction_save_path, exist_ok=True)

        self.model = super_network
        self.run_config = run_config
        self.out_log = out_log

        self.best_monitor = self.build_monitor(self.run_config.monitor)
        self.start_epoch = 0

        # initialize model
        self.model.init_model(self.run_config.model_init, self.run_config.init_div_groups)
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

    ''' net info '''
    def net_flops(self):
        # TODO: get flops of specific architecture, related to architecture parameter

        data_shape = [1] + list(self.run_config.data_provider.data_shape)
        # data_shape [1, 3, 512, 512]

        net = self.net
        #
        input_var = torch.zeros(data_shape, device=self.device)
        with torch.no_grad():
            flop, _ = net.get_flops(input_var)
        return flop

    ''' save and load models '''
    def save_model(self, epoch, checkpoint=None, is_warmup=False, is_best=False, checkpoint_file_name=None):

        # what need to save
        # 1. network state_dict                       √
        # 2. weight_optimizer state_dict              √
        # 3. arch_optimizer state_dict                √ √
        # 4. weight_scheduler state_dict              √
        # 5. arch_scheduler state_dict if is not None √ √
        # 6. epochs, or start_epochs                  √
        # 7. warmup or not                            √
        # 8. best_monitor and best_value              √
        # 9. the best one, and per frequency one      √

        # need modifications
        if checkpoint_file_name is None:
            if is_warmup:
                checkpoint_file_name = 'checkpoint-warmup.pth.tar'
            else:
                if is_best: checkpoint_file_name = 'checkpoint-best.pth.tar'
                else: checkpoint_file_name = 'checkpoint.pth.tar'

        if checkpoint is not None:
            # not None, used in nas_manager, when warmup, train search.
            # can offer warmup, arch_optimizer information
            if is_warmup:
                # in warmup phase, need save

                save_path = os.path.join(self.ckpt_save_path, 'warmup')
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, checkpoint_file_name)
            else:
                checkpoint.update({
                    'state_dict': self.model.state_dict(),
                    'weight_optimizer': self.optimizer.state_dict(),
                    'weight_scheduler': self.scheduler.state_dict(),
                    'best_monitor': (self.monitor_metric, self.best_monitor),
                    'warmup': is_warmup,
                    'start_epochs': epoch + 1,
                })
                save_path = os.path.join(self.ckpt_save_path, 'train_search')
                os.makedirs(save_path, exist_ok=True)
                save_path = os.path.join(save_path, checkpoint_file_name)
        else: # checkpoint is None in only train phase, in self.train()

            state_dict = self.model.state_dict()
            #for key in list(state_dict.keys()):
            #    if 'cell_arch_parameters' in key or 'network_arch_parameters' in key or 'aspp_arch_parameters' in key:
            #        state_dict.pop(key)
            checkpoint = {
                'state_dict': state_dict,
                'weight_optimizer': self.optimizer.state_dict(),
                'weight_scheduler': self.scheduler.state_dict(),
                'best_monitor': (self.monitor_metric, self.best_monitor),
                'start_epochs': epoch + 1,
            }
            save_path = os.path.join(self.ckpt_save_path, 'retrain')
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path, checkpoint_file_name)
        torch.save(checkpoint, save_path)


    def load_model(self, checkpoint_file):
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

    def write_log(self, log_str, prefix, should_print=True):
        # needs three types logs,
        # 1. train_search_log
        # 2. validation_log
        # 3. testing_log
        # 4. arch_log, records arch_information
        # 5. retrain log
        assert prefix in ['train_search', 'validation', 'testing', 'arch', 'train', 'warmup']
        with open(os.path.join(self.log_save_path, prefix + '.txt'), 'a') as fout:
            fout.write('=' * 10 + '\n')
            fout.write(log_str + '\n')
            fout.flush()
        if should_print:
            print(log_str)

    def validate(self, epoch, is_test=False, use_train_mode=False):
        #
        # TODO: test and validate should both use the derived model.
        #
        # 1. super network viterbi_decodde, get actual_path
        # 2. cells genotype decode, which are on the actual_path in the super network
        # 3. according to actual_path and cells genotypes, construct the best network.
        # 4. use the best network, to perform test phrase.

        if is_test: data_loader = self.run_config.test_loader
        else: data_loader = self.run_config.valid_loader
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
            for i, (datas, targets) in enumerate(data_loader):
                if torch.cuda.is_available():
                    datas = datas.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)
                else:
                    raise ValueError('do not support cpu version')
                data_time.update(time.time()-end0)


                # TODO: need modification, in validation and testing phrase, forward in derived model.
                logits = model.single_path_forward(datas)

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

            if is_test: prefix = 'testing'
            else: prefix = 'validation'
            use_time = 'Time Duration: {:}, Data Time : {:}'\
                .format(convert_secs2time(batch_time.average, True), convert_secs2time(data_time.average, True))
            epoch_str = '{:03d}-{:03d}'.format(epoch, self.run_config.total_epochs)
            log_str = '[{:}] {:} : loss={:.2f}, accuracy={:.2f}, miou={:.2f}, f1score={:.2f}'\
                .format(epoch_str, prefix, losses.average, accs.average, mious.average, fscores.average)
            log_str = use_time+'\n'+log_str
            self.write_log(log_str, prefix)
        return losses.avg, accs.avg, mious.avg, fscores.avg

    def train_one_epoch(self, epoch, train_log_func):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        mious = AverageMeter()
        fscores = AverageMeter()

        self.model.train()

        self.scheduler.step(epoch)
        train_lr = self.scheduler.get_lr()
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
            if (i+1) % self.run_config.train_print_freq or (i+1) == len(self.run_config.train_loader):
                batch_log = train_log_func(i, batch_time, data_time, losses, accs, mious, fscores, train_lr)
                # write log and print for training
                self.write_log(batch_log, 'train', should_print=True)
        return losses.avg, accs.avg, mious.avg, fscores.avg

    def train(self):
        iter_per_epoch = len(self.run_config.train_loader)
        def train_log_func(epoch_, i, batch_time, data_time, losses, accs, mious, fscores, new_lr):
            time_per_iter = batch_time.average
            time_left = self.run_config.total_epochs * iter_per_epoch - (epoch_*iter_per_epoch + i+1) *time_per_iter
            epoch_str = '|iter[{:03d}/{:03d}]-epoch[{:03d}/{:03d}]|'.format(i+1, iter_per_epoch, epoch_+1, self.run_config.total_epochs)
            common_log = '[Training the {:}] Time={:}/iter Left={:} LR={:}'\
                .format(epoch_str, time_per_iter, time_left, new_lr)
            #time_log =  'Time Use : {:}, Data Time : {:}'\
            #    .format(convert_secs2time(batch_time.average, True), convert_secs2time(data_time.average, True))
            batch_log = '[{:}] training : loss={:.2f} accuracy={:.2f} miou={:.2f} f1score={:.2f}'\
                .format(epoch_str, losses.average, accs.average, mious.average, fscores.average)
            batch_log = common_log+'\n'+batch_log
            return batch_log

        # TODO: in retrain phase, should modify total_epochs
        for epoch in range(self.start_epoch, self.run_config.total_epochs):
            logging.info('\n'+'-'*30, 'Train epoch: {}'.format(epoch+1), '-'*30+'\n')
            # train log have been wrote in train_one_epoch
            loss, acc, miou, fscore = self.train_one_epoch(epoch,
                lambda i, batch_time, data_time, losses, accs, mious, fscores, new_lr: train_log_func(
                    epoch, i, batch_time, data_time, losses, accs, mious, fscores, new_lr
                ))

            #time_per_epoch = time.time() - end
            #seconds_left = int((self.run_config.total_epochs - 1 - epoch) * time_per_epoch)
            #print('Time per epoch: {}, Est. complete in: {}'
            #             .format(str(timedelta(seconds=time_per_epoch)),
            #                     str(timedelta(seconds=seconds_left))))
            # perform validation at the end of each epoch.
            if (epoch+1) % self.run_config.validation_freq == 0 or (epoch+1) == self.run_config.total_epochs:
                # have write validation_log in self.validate
                val_loss, val_acc, val_miou, val_fscore = self.validate(epoch, is_test=False, use_train_mode=False)
                val_monitor_metric = get_monitor_metric(self.monitor_metric, val_loss, val_acc, val_miou, val_fscore)
                is_best = val_monitor_metric > self.best_monitor
                self.best_monitor = max(self.best_monitor, val_monitor_metric)
            else:
                is_best = False

            if (epoch+1) % self.run_config.save_ckpt_freq == 0 or (epoch+1) == self.run_config.total_epochs:
                # re-train, only have weight_optimizer
                # checkpoint is None, by default
                self.save_model(epoch, is_warmup=False, is_best=is_best)

