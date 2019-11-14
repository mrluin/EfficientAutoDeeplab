import time
import os
import math
import torch
import logging
import torch.nn as nn
import json

from datetime import timedelta
from data.WHUBuilding import WHUBuildingDataProvider
from utils.common import set_manual_seed
from utils.common import get_monitor_metric
from utils.common import AverageMeter
from utils.common import count_parameters
from utils.loss import SegmentationLosses
from utils.metrics import Evaluator
from utils.calculators import calculate_weights_labels


'''
# RunConfig: 1. all the configurations from args
#            2. build optimizer, learning_rate, and dataset
#
# RunManager: 1. manage to train, valid and test
#             2. processes related to training phrase
'''

class RunConfig:
    def __init__(self, total_epochs,
                 gpu_ids, workers,
                 save_path, dataset, nb_classes, train_batch_size, valid_size, test_batch_size,
                 ori_size, crop_size,
                 init_lr, lr_scheduler, lr_scheduler_param,
                 optim_type, optim_params, weight_decay,
                 label_smoothing, no_decay_keys,
                 model_init, init_div_groups, filter_multiplier, block_multiplier, steps, bn_momentum, bn_eps, dropout, nb_layers,
                 validation_freq, print_freq, save_ckpt_freq, monitor, print_save_arch_information, save_normal_net_after_training,
                 print_arch_param_step_freq,
                 use_unbalanced_weights,
                 **kwargs):

        self.total_epochs = total_epochs

        self.gpu_ids = gpu_ids
        self.workers = workers

        self.save_path = save_path
        self.dataset = dataset
        self.nb_classes = nb_classes
        self.train_batch_size = train_batch_size
        self.valid_size = valid_size
        self.test_batch_size = test_batch_size

        self.ori_size = ori_size
        self.crop_size = crop_size

        self.init_lr = init_lr
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_param = lr_scheduler_param

        self.optim_type = optim_type
        self.optim_params = optim_params
        self.weight_decay = weight_decay

        self.label_smoothing = label_smoothing
        self.no_decay_keys = no_decay_keys

        # TODO: add network config
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
        self.print_freq = print_freq
        self.save_ckpt_freq = save_ckpt_freq
        self.monitor = monitor
        self.print_save_arch_information = print_save_arch_information
        self.save_normal_net_after_training = save_normal_net_after_training
        self.print_arch_param_step_freq = print_arch_param_step_freq

        self.use_unbalanced_weights = use_unbalanced_weights

        self._data_provider = None
        self._train_iter, self._valid_iter, self._test_iter = None, None, None
        self.optimizer = None
    @property
    def config(self):
        config = {
            'type': type(self)
        }
        for key in self.__dict__:
            if not key.startswith('_'):
                config[key] = self.__dict__[key]
        return config

    def copy(self):
        return RunConfig(**self.config)

    ''' learning rate '''
    def calc_learning_rate(self, epoch, iteration=0, iter_per_epoch=None, lr_max=None, warmup_lr=False):
        '''
            step mode: lr = base_lr * 0.1 ^ (floor(epoch-1/lr_step))
            cosine mode: lr = base_lr * 0.5 * (1+cos(iter/maxiter))
            poly mode: lr = base_lr * (1-iter/maxiter)^0.9

            from args
            :attr: self.lr_scheduler
            :attr: self.init_lr
            :attr: self.total_epochs

            :attr: iteration
            :attr: iter_per_epoch
            :attr: lr_max, warmup_lr
            # TODO poly and step need lr_scheduler_params
            # TODO min_lr
        '''
        total_iter = self.total_epochs * iter_per_epoch
        current_iter = epoch * iter_per_epoch + iteration
        if warmup_lr and lr_max is not None:
            lr = 0.5 * lr_max * (1 + math.cos(math.pi * current_iter / total_iter))
        else:

            if self.lr_scheduler == 'cosine':
                lr = self.init_lr * 0.5 * (1 + math.cos(math.pi * current_iter / total_iter))
            elif self.lr_scheduler == 'poly':
                #lr = self.init_lr * pow((1 - (iteration - self.warmup_iters) / (total_iter - self.warmup_iters)), 0.9)
                raise NotImplementedError
            elif self.lr_scheduler == 'step':
                raise NotImplementedError
            else:
                raise NotImplementedError
        return lr

    def adjust_learning_rate(self, optimizer, epoch, iteration, iter_per_epoch=None, lr_max=None, warmup_lr=False):
        new_lr = self.calc_learning_rate(epoch, iteration, iter_per_epoch, lr_max, warmup_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr
    '''
    @property
    def learning_rate(self):
        return self.optimizer.param_groups[0]['lr']
        '''

    ''' data provider '''
    @property
    def data_config(self):
        # need             save_path,
        #                  train_batch_size,
        #                  valid_size,
        #                  test_batch_size,
        #                  nb_works
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
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
    def true_valid_loader(self):
        return self.data_provider.true_valid_loader

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

    ''' optimizer '''
    def build_optimizer(self, net_params):
        '''
        :param net_params: len(net_params) == 2, net_params[0] with weight_decay, net_params[1] without weight_decay
        '''
        if self.optim_type == 'sgd':
            optim_params = {} if self.optim_params is None else self.optim_params
            momentum, nesterov = optim_params.get('momentum', 0.9), optim_params.get('nesterov', True)
            if self.no_decay_keys:
                self.optimizer = torch.optim.SGD([
                    {'params': net_params[0], 'weight_decay': self.weight_decay},
                    {'params': net_params[1], 'weight_decay': 0}
                ], self.init_lr, momentum=momentum, nesterov=nesterov)
            else:
                self.optimizer = torch.optim.SGD(net_params, self.init_lr, momentum, weight_decay=self.weight_decay, nesterov=nesterov)
        else: raise NotImplementedError

        return self.optimizer

class RunManager:

    def __init__(self, path, net, run_config: RunConfig, out_log=True, measure_latency=None):

        # logs have

        self.path = path
        self._save_path = None # path to checkpoint file
        self._log_path = None # path to logs file
        self._prediction_path = None # path to predictions file

        self.net = net
        self.run_config = run_config
        self.out_log = out_log

        self.best_monitor = self.build_monitor(self.run_config.monitor)
        self.start_epoch = 0

        # initialize model
        # TODO model.init_model
        self.net.init_model(self.run_config.model_init, self.run_config.init_div_groups)

        # a copy of net on cpu for latency estimation & mobile latency

        # move network to GPU if available
        # use single gpu to train by default
        if torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(self.run_config.gpu_ids))
            print(self.device)
            self.net.to(self.device)
        else:
            raise ValueError('do not support cpu version')

        # TODO: print net info
        # self.print_net_info(measure_latency)

        # create loss function
        classes_weight = None
        if self.run_config.use_unbalanced_weights:
            classes_weight = calculate_weights_labels(self.path, self.run_config.dataset, self.run_config.train_loader,
                                                      self.run_config.nb_classes)
        label_smoothing = self.run_config.label_smoothing
        #self.criterion = SegmentationLosses(classes_weight, label_smoothing, cuda=self.run_config.gpu_ids)
        # TODO: modification in self.criterion
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        if self.run_config.no_decay_keys:
            keys = self.run_config.no_decay_keys.split('#')
            self.optimizer = self.run_config.build_optimizer([
                # TODO get two groups of parameters according to keys
            ])
        else:
            self.optimizer = self.run_config.build_optimizer(self.net.weight_parameters())
        #self.optimizer.cuda()
        # TODO:
        #self.optimizer = self.optimizer.to(self.device)

    ''' save path and log path '''
    @property
    def save_path(self):
        if self._save_path is None:
            save_path = os.path.join(self.path, 'checkpoint')
            os.makedirs(save_path, exist_ok=True)
            self._save_path = save_path
        return self._save_path

    @property
    def log_path(self):
        if self._log_path is None:
            log_path = os.path.join(self.path, 'logs')
            os.makedirs(log_path, exist_ok=True)
            self._log_path = log_path
        return self._log_path

    @property
    def prediction_path(self):
        if self._prediction_path is None:
            prediction_path = os.path.join(self.path, 'predictions')
            os.makedirs(prediction_path, exist_ok=True)
            self._prediction_path = prediction_path
        return self._prediction_path

    def build_monitor(self, monitor):
        monitor_mode = monitor.split('#')[0]
        self.monitor_metric = monitor.split('#')[1]
        assert self.monitor_metric in ['miou', 'fscore'], 'invalid monitor metric'
        best_monitor = math.inf if monitor_mode == 'min' else -math.inf
        return best_monitor

    ''' net info '''
    def net_flops(self):
        # arch_network_parameter, can only get expected flops
        # TODO: get flops of specific architecture, related to architecture parameter

        data_shape = [1] + list(self.run_config.data_provider.data_shape)
        # data_shape [1, 3, 512, 512]

        net = self.net
        input_var = torch.zeros(data_shape, device=self.device)
        with torch.no_grad():
            flop, _ = net.get_flops(input_var)
        return flop


    def net_latency(self):
        raise NotImplementedError

    def print_net_info(self, measure_latency=None):
        # network architecture
        if self.out_log:
            #logging.info(self.net)
            print(self.net)
        # parameters
        if isinstance(self.net, nn.DataParallel):
            raise NotImplementedError
        else:
            total_params = count_parameters(self.net)

        if self.out_log:
            #logging.info('Total training params: {:.2f}M'.format(total_params / 1e6))
            print('Total Training Params: {:.2f}M'.format(total_params / 1e6))
        net_info = {
            'param': '{:.2f}M'.format(total_params / 1e6)
        }

        # TODO: flops
        #flops = self.net_flops()
        #if self.out_log:
            #logging.info('Total FLOPs: {:.2f}M'.format(flops / 1e6))
        #    print('Total Training FLOPs: {:.2f}M'.format(flops / 1e6))
        #net_info['flops'] = '{:.2f}M'.format(flops / 1e6)

        # TODO: latency constraint
        # not implement
        # write net_info logs
        with open('{}/net_info.txt'.format(self.log_path), 'w') as fout:
            fout.write(json.dumps(net_info, indent=4) + '\n')

    ''' save and load models '''
    def save_model(self, epoch, checkpoint=None, is_best=False, checkpoint_file_name=None):

        # when to save self.run_config.save_freq
        # :epoch: √
        # :checkpoint:
        # :is_best: √
        # :checkpoint_file_name:
        assert checkpoint is not None, 'checkpoint is None'
        '''
        if checkpoint is None:
            checkpoint = {'state_dict', self.net.state_dict()}
            # TODO: confirm, parallel or not
            # self.net.module.state_dict() or self.net.state_dict()
            '''
        if checkpoint_file_name is None:
            checkpoint_file_name = 'checkpoint-{}.pth.tar'.format(epoch)

        checkpoint['dataset'] = self.run_config.dataset
        # other information has been included in checkpoint

        latest_fname = os.path.join(self.save_path, 'latest.txt') # record saved checkpoint_file
        model_path = os.path.join(self.save_path, checkpoint_file_name)
        with open(latest_fname, 'w') as fout:
            fout.write(model_path + '\n')

        torch.save(checkpoint, model_path)

        if is_best:
            best_path = os.path.join(self.save_path, 'checkpoint-best.pth.tar')
            #torch.save({'state_dict': checkpoint['state_dict']}, best_path)
            torch.save(checkpoint, best_path)

    def load_model(self, ckptfile_path=None):

        assert ckptfile_path is not None and os.path.exists(ckptfile_path),\
            'checkpoint_file can not find'

        '''   
        latest_fname = os.path.join(self.save_path, 'latest.txt')
        # if not point to specific checkpoint_file
        if ckpt_filename is None and os.path.exists(latest_fname):
            with open(latest_fname, 'r') as fin:
                ckpt_filename = fin.readline()
                if ckpt_filename[-1] == '\n':
                    ckpt_filename = ckpt_filename[:-1]
        # get the latest one, according to latest.txt
        '''
        try:
            '''
            if ckpt_filename is None or not os.path.exists(ckpt_filename):
                # TODO: this case has issue
                ckpt_filename = '{}/checkpoint.pth.tar'.format(self.save_path)
                '''
            if self.out_log:
                #logging.info('Loading Checkpoint {}'.format(ckptfile_path))
                print('='*30+'=>\tLoading Checkpoint {}'.format(ckptfile_path))

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
            if 'best_{}'.format(self.monitor_metric) in checkpoint:
                self.best_monitor = checkpoint['best_{}'.format(self.monitor_metric)]
            # optimizer for only training, weight_optimizer and arch_optimizer for train_search phrase
            if 'optimizer' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.out_log:
                #logging.info('Loaded Checkpoint {}'.format(ckptfile_path))
                print('='*30+'=>\tLoaded Checkpoint {}'.format(ckptfile_path))
        except Exception:
            if self.out_log:
                #logging.info('Fail to load Checkpoint {}'.format(ckptfile_path))
                print('='*30+'=>\tFail to load Checkpoint {}'.format(ckptfile_path))

    def save_config(self, print_info=True):
        # dumps run_config and net_config to model_folder
        os.makedirs(self.path, exist_ok=True)
        net_config_save_path = os.path.join(self.path, 'net.config')
        # TODO: net.config
        # net_config, related to network architecture
        # and net_info, parameters, flops, and latency
        json.dump(self.net.config, open(net_config_save_path, 'w'), indent=4)
        if print_info:
            #logging.info('Network Configs dumps to {}'.format(net_config_save_path))
            print('='*30,'Network Configs dumps to {}'.format(net_config_save_path))

        run_config_save_path = os.path.join(self.path, 'run.config')
        json.dump(self.run_config.config, open(run_config_save_path, 'w'), indent=4)
        if print_info:
            #logging.info('Run Configs dumps to {}'.format(run_config_save_path))
            print('='*30, 'Run Configs dumps to {}'.format(run_config_save_path))

    ''' train and test '''
    def write_log(self, log_str, prefix, should_print=True):
        assert prefix in ['gradient_search', 'arch', 'train', 'valid', 'test'], 'invalid prefix'
        # train log
        # valid log
        # test log
        # gradient_search log
        # arch log
        if prefix in ['gradient_search', 'arch']:
            with open(os.path.join(self.log_path, prefix+'.txt'), 'a') as fout:
                fout.write(log_str)
                fout.flush()
        # valid + test
        if prefix in ['valid', 'test']:
            with open(os.path.join(self.log_path, 'valid_test_console.txt'), 'a') as fout:
                if prefix == 'test':
                    fout.write('=' * 10 + '\n')
                    fout.write(log_str + '\n')
                    fout.write('=' * 10 + '\n')
                fout.write(log_str + '\n')
                fout.flush()

        # train + valid + test
        # train re-confirmed
        if prefix in ['valid', 'test', 'train']:
            with open(os.path.join(self.log_path, 'train_console.txt'), 'a') as fout:
                if prefix in ['valid', 'test']:
                    fout.write('=' * 10 + '\n')
                    fout.write(log_str + '\n')
                    fout.write('=' * 10 + '\n')
                fout.write(log_str + '\n')
                fout.flush()
        if should_print:
            #logging.info(log_str)
            print(log_str)

    def validate(self, is_test=False, net=None, use_train_mode=False):
        if is_test:
            data_loader = self.run_config.test_loader
        else:
            data_loader = self.run_config.valid_loader
        if net is None:
            net = self.net
        if use_train_mode:
            net.train()
        else:
            net.eval()

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

                logits = net(datas)
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

                if i % self.run_config.print_freq == 0 or i + 1 == len(data_loader):
                    if is_test:
                        prefix = 'Test'
                    else:
                        prefix = 'Valid'
                    test_log = prefix + '\t[{0}/{1}]\n' \
                                        'Time\t{batch_time.val:.4f}\t({batch_time.avg:.4f})\n' \
                                        'Data\t{data_time.val:.4f}\t({data_time.avg:.4f})\n' \
                                        'Loss\t{loss.val:.6f}\t({loss.avg:.6f})\n' \
                                        'Acc\t{acc.val:6.4f}\t({acc.avg:6.4f})\n' \
                                        'mIoU\t{mIoU.val:6.4f}\t({mIoU.avg:6.4f})\n' \
                                        'F1\t{F1.val:6.4f}\t({F1.avg:6.4f})\n'\
                        .format(i, len(data_loader)-1, batch_time=batch_time, data_time=data_time,
                                loss=losses, acc=accs, mIoU=mious, F1=fscores)
                    # TODO: write test or valid logs
                    #logging.info(test_log)
                    #self.write_log(test_log, prefix=prefix, should_print=True)

        return losses.avg, accs.avg, mious.avg, fscores.avg

    def train_one_epoch(self, adjust_lr_func, train_log_func):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        mious = AverageMeter()
        fscores = AverageMeter()

        self.net.train()

        end = time.time()
        for i, (datas, targets) in enumerate(self.run_config.train_loader):
            if torch.cuda.is_available():
                datas = datas.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
            else:
                raise ValueError('do not support cpu version')
            data_time.update(time.time()-end)
            new_lr = adjust_lr_func(i)

            logits = self.net(datas)
            # do not use label_smoothing by default
            if self.run_config.label_smoothing > 0:
                raise NotImplementedError
            else:
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
            # TODO: self.net.zero_grad() or self.optimizer.zero_grad()
            #self.optimizer.zero_grad()
            self.net.zero_grad()
            loss.backward()
            self.optimizer.step()
            # elapsed time
            batch_time.update(time.time()-end)
            end = time.time()

            # within one epoch, per iteration print train_log
            if i % self.run_config.print_freq or (i+1) == len(self.run_config.train_loader):
                batch_log = train_log_func(i, batch_time, data_time, losses, accs, mious, fscores, new_lr)

                # write log and print for training
                self.write_log(batch_log, 'train', should_print=True)

        return losses.avg, accs.avg, mious.avg, fscores.avg

    def train(self):
        iter_per_epoch = len(self.run_config.train_loader)
        def train_log_func(epoch_, i, batch_time, data_time, losses, accs, mious, fscores, new_lr):
            batch_log = 'Train\t[{0}][{1}/{2}]\tlr {lr:.5f}\n' \
                        'Time\t{batch_time.val:.4f}\t({batch_time.avg:.4f})\n' \
                        'Data\t{data_time.val:.4f}\t({data_time.avg:.4f})\n' \
                        'Loss\t{losses.val:.6f}\t({losses.avg:.6f})\n' \
                        'Acc\t{accs.val:6.4f}\t({accs.avg:6.4f})\n' \
                        'mIoU\t{miou.val:6.4f}\t({miou.avg:6.4f})\n' \
                        'F1\t{fscore.val:6.4f}\t({fscore.avg:6.4f})\n'\
                .format(epoch_+1, i, iter_per_epoch-1, lr=new_lr, batch_time=batch_time, data_time=data_time,
                    losses=losses, accs=accs, miou=mious, fscore=fscores)
            return batch_log

        for epoch in range(self.start_epoch, self.run_config.total_epochs):
            logging.info('\n', '-'*30, 'Train epoch: {}'.format(epoch+1), '-'*30, '\n')

            end = time.time()
            # one epoch training process, rt mean value
            loss, acc, miou, fscore = self.train_one_epoch(
                lambda i: self.run_config.adjust_learning_rate(self.optimizer, epoch, i, iter_per_epoch=iter_per_epoch),
                lambda i, batch_time, data_time, losses, accs, mious, fscores, new_lr: train_log_func(
                    epoch, i, batch_time, data_time, losses, accs, mious, fscores, new_lr
                ))
            time_per_epoch = time.time() - end
            seconds_left = int((self.run_config.total_epochs - 1 - epoch) * time_per_epoch)
            print('Time per epoch: {}, Est. complete in: {}'
                         .format(str(timedelta(seconds=time_per_epoch)),
                                 str(timedelta(seconds=seconds_left))))
            if (epoch + 1) % self.run_config.validation_freq == 0:
                val_loss, val_acc, val_miou, val_fscore = self.validate(is_test=False, net=self.net, use_train_mode=False,)
                val_monitor_metric = get_monitor_metric(self.monitor_metric, val_loss, val_acc, val_miou, val_fscore)
                is_best = val_monitor_metric > self.best_monitor
                self.best_monitor = max(self.best_monitor, val_monitor_metric)
                # val_log: combination of valid log and training log within a same epoch
                val_log = 'Valid\t[{0}/{1}]\n' \
                          'Loss\t{2:.6f}\tAcc\t{3:6.4f}\tmIoU\t{4:6.4f}\tF1\t{5:6.4f}\n'\
                    .format(epoch+1, self.run_config.total_epochs, val_loss, val_acc, val_miou, val_fscore)
                val_log += 'Train\t[{0}/{1}]\n' \
                           'Loss\t{2:.6f}\tAcc\t{3:6.4f}\tmIoU\t{4:6.4f}\tF1\t{5:6.4f}\n'\
                    .format(epoch+1, self.run_config.total_epochs, loss, acc, miou, fscore)

                # write and print validation log
                self.write_log(val_log, 'valid', should_print=True)
            else:
                is_best = False

            if (epoch + 1) % self.run_config.save_ckpt_freq == 0:
                # pass epoch, checkpoint, and is_best
                self.save_model(epoch, {
                    'epoch': epoch,
                    'best_{}'.format(self.monitor_metric): self.best_monitor,
                    'optimizer': self.optimizer.state_dict(),
                    'state_dict': self.net.state_dict(),
                }, is_best=is_best)
