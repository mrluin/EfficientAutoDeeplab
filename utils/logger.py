'''
@author: Jingbo Lin
@contact: ljbxd180612@gmail.com
@github: github.com/mrluin
'''
from os import path as osp
from pathlib import Path
import importlib, warnings
import os, sys, time, numpy as np
from io import BytesIO as BIO
from shutil import copyfile

import visdom
from copy import deepcopy
import PIL
import torch

def prepare_logger(args):
    args = deepcopy(args)
    # TODO: add log_dir and random_seed
    #log_dir = os.path.join(args.path, 'logs')
    logger = Logger(args.path ,args.random_seed)
    logger.log('Main Function with logger : {:}'.format(logger), mode='info')
    logger.log('Arguments : ------------------------------------', mode='info')
    for name, value in args._get_kwargs():
        logger.log('{:16} : {:}'.format(name, value), mode='info')
    logger.log("Python  Version  : {:}".format(sys.version.replace('\n', ' ')), mode='info')
    logger.log("Pillow  Version  : {:}".format(PIL.__version__), mode='info')
    logger.log("PyTorch Version  : {:}".format(torch.__version__), mode='info')
    logger.log("cuDNN   Version  : {:}".format(torch.backends.cudnn.version()), mode='info')
    logger.log("CUDA available   : {:}".format(torch.cuda.is_available()), mode='info')
    logger.log("CUDA GPU numbers : {:}".format(torch.cuda.device_count()), mode='info')
    logger.log("CUDA_VISIBLE_DEVICES : {:}".format(
        os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'None'), mode='info')
    return logger

class Logger(object):
    def __init__(self, path, seed, create_model_dir=True, create_prediction_dir=True,
                 use_tf=False):

        self.seed = int(seed)
        self.log_dir = Path(path) / 'logs'
        self.model_dir = Path(path) / 'checkpoints'
        self.predictions_dir = Path(path) / 'predictions'

        self.log_dir.mkdir(parents=True, exist_ok=True)
        if create_model_dir:
            self.model_dir.mkdir(parents=True, exist_ok=True)
        if create_prediction_dir:
            self.predictions_dir.mkdir(parents=True, exist_ok=True)
        '''
        # TODO: to use visdom rather than tensorboardX
        self.use_tf = bool(use_tf)
        self.tensorboard_dir = self.log_dir / ('tensorboard-{:}'.format(time.strftime('%d-%h', time.gmtime(time.time()))))
        if self.use_tf:
            self.tensorboard_dir.mkdir(mode=0o775, parents=True, exist_ok=True)
            self.writer = tf.summary.FileWriter(str(self.tensorboard_dir))
        else:
            self.writer = None
            '''
        self.logger_path_info = self.log_dir / 'seed-{:}-T-{:}-info.log'.format(self.seed, time.strftime('%d-%h-at-%H-%M-%S', time.gmtime(time.time())))
        self.logger_path_warm = self.log_dir / 'seed-{:}-T-{:}-warm.log'.format(self.seed, time.strftime('%d-%h-at-%H-%M-%S', time.gmtime(time.time())))
        self.logger_path_search = self.log_dir / 'seed-{:}-T-{:}-search.log'.format(self.seed, time.strftime('%d-%h-at-%H-%M-%S', time.gmtime(time.time())))
        self.logger_path_retrain = self.log_dir / 'seed-{:}-T-{:}-retrain.log'.format(self.seed, time.strftime('%d-%h-at-%H-%M-%S', time.gmtime(time.time())))
        self.logger_path_valid = self.log_dir / 'seed-{:}-T-{:}-valid.log'.format(self.seed, time.strftime('%d-%h-at-%H-%M-%S', time.gmtime(time.time())))
        self.logger_path_test = self.log_dir / 'seed-{:}-T-{:}-test.log'.format(self.seed, time.strftime('%d-%h-at-%H-%M-%S', time.gmtime(time.time())))

        self.logger_file_info = open(self.logger_path_info, 'w')
        self.logger_file_warm = open(self.logger_path_warm, 'w')
        self.logger_file_search = open(self.logger_path_search, 'w')
        self.logger_file_retrain = open(self.logger_path_retrain, 'w')
        self.logger_file_valid = open(self.logger_path_valid, 'w')
        self.logger_file_test = open(self.logger_path_test, 'w')


    def __repr__(self):
        return ('{name}(dir={log_dir}), (ckpt_dir={model_dir}), (prediction_dir={predictions_dir})'.format(name=self.__class__.__name__, **self.__dict__))



    def path(self, mode, is_best=False):
        valids = ('warm', 'search', 'retrain')
        if   mode == 'warm'   : return self.model_dir / 'seed-{:}-warm.pth'.format(self.seed)
        elif mode == 'search' : return self.model_dir / 'seed-{:}-search.pth'.format(self.seed) if is_best==False else self.model_dir / 'seed-{:}-search-best.pth'.format(self.seed)
        elif mode == 'retrain': return self.model_dir / 'seed-{:}-retrain.pth'.format(self.seed) if is_best==False else self.model_dir / 'seed-{:}-retrain.pth'.format(self.seed)
        else: raise TypeError('Unknow mode = {:}, valid modes = {:}'.format(mode, valids))

    def extract_log(self, mode):
        if mode == 'warm':
            return self.logger_file_warm
        elif mode == 'search':
            return self.logger_file_search
        elif mode == 'retrain':
            return self.logger_file_retrain
        elif mode == 'valid':
            return self.logger_file_valid
        elif mode == 'test':
            return self.logger_file_test
        elif mode == 'info':
            return self.logger_file_info
        else:
            raise NotImplementedError

    def close(self):
        self.logger_file_warm.close()
        self.logger_file_search.close()
        self.logger_file_retrain.close()
        self.logger_file_valid.close()
        self.logger_file_test.close()
        self.logger_file_info.close()

    def log(self, string, mode, save=True, stdout=False):
        if stdout:
            sys.stdout.write(string); sys.stdout.flush()
        else:
            print(string)
        if save:
            if mode == 'warm':
                self.logger_file_warm.write('{:}\n'.format(string))
                self.logger_file_warm.flush()
            elif mode == 'search':
                self.logger_file_search.write('{:}\n'.format(string))
                self.logger_file_search.flush()
            elif mode == 'retrain':
                self.logger_file_retrain.write('{:}\n'.format(string))
                self.logger_file_retrain.flush()
            elif mode == 'valid':
                self.logger_file_valid.write('{:}\n'.format(string))
                self.logger_file_valid.flush()
            elif mode == 'test':
                self.logger_file_test.write('{:}\n'.format(string))
                self.logger_file_test.flush()
            elif mode == 'info':
                self.logger_file_info.write('{:}\n'.format(string))
                self.logger_file_info.flush()
            else:
                ValueError('do not support mode {:}'.format(mode))

    '''
        # visualization method change into visom
    def scalar_summary(self, tags, values, step):
        if not self.use_tf:
            warnings.warn('do set use-tensorflow installed but call scalar_summary')
        else:
            assert isinstance(tags, list) == isinstance(values, list), 'Type: {:} vs. {:}'.format(type(tags), type(values))
            if not isinstance(tags, list):
                tags, values = [tags], [values]
            for tag, value in zip(tags, values):
                summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)
                self.wirter.flush()

    def image_summary(self, tag, images, step):
        import scipy
        if not self.use_tf:
            warnings.warn('do set use-tensorflow installed but call scalar_summary')
            return

        img_summaries = []
        for i, img in enumerate(images):
            # write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format='png')

            # create an image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # create a summary value
            img_summaries.append(tf.Summary.Value(tag='{}/{}'.format(tag, i), image=img_sum))

        # create and write summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        if not self.use_tf: raise ValueError('do not have tensorflow')
        import tensorflow as tf

        # create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # drop the start of the first bin
        bin_edges = bin_edges[1:]

        # add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # create and write summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
    '''
def save_checkpoint(state, filename, logger, mode):
    if osp.isfile(filename):
        if hasattr(logger, 'log'):
            logger.log('Find {:} exist, delete it at first before saving'.format(filename), mode)
        os.remove(filename)
    torch.save(state, filename)
    assert osp.isfile(filename), 'save filename : {:} failed, which is not found.'.format(filename)
    if hasattr(logger, 'log'):
        logger.log('save checkpoint into {:}'.format(filename), mode)

    return filename

def copy_checkpoint(src, dst, logger, mode):
    if osp.isfile(dst):
        if hasattr(logger, 'log'):
            logger.log('Find {:} exist, delete it at the first before saving'.format(dst), mode)
        os.remove(dst)
    copyfile(src, dst)
    if hasattr(logger, 'log'): logger.log('copy the file from {:} into {:}'.format(src, dst), mode)

def display_all_families_information(args, nas_manager, logger):
    log_str = '==================== {:10s} ====================\n'.format(nas_manager.run_manager.run_config.dataset)
    log_str +='Train Loader :: Len={:} batch_size={:}\n'.format(len(nas_manager.run_manager.run_config.train_loader), nas_manager.run_manager.run_config.train_loader.batch_size)
    log_str +='Valid Loader :: Len={:} batch_size={:}\n'.format(len(nas_manager.run_manager.run_config.valid_loader), nas_manager.run_manager.run_config.valid_loader.batch_size)
    log_str +='Test  Loader :: Len={:} batch_size={:}\n'.format(len(nas_manager.run_manager.run_config.test_loader), nas_manager.run_manager.run_config.test_loader.batch_size)
    log_str +='==================== OPTIMIZERS ====================\n'
    log_str +='weight_optimizer :: {:}\n'.format(nas_manager.run_manager.optimizer)
    log_str +='arch_optimizer   :: {:}\n'.format(nas_manager.arch_optimizer)
    log_str +='weight_scheduler :: {:} T_max={:} eta_min={:}\n'.format(args.scheduler, args.T_max, args.eta_min)
    log_str +='criterion       :: {:}\n'.format(nas_manager.run_manager.criterion)
    log_str +='==================== SearchSpace ====================\n'
    log_str += str(args.conv_candidates) + '\n'
    log_str +='==================== SuperNetwork Config ====================\n'
    log_str +='filter_multiplier :: {:}\n'.format(args.filter_multiplier)
    log_str +='block_multiplier  :: {:}\n'.format(args.block_multiplier)
    log_str +='steps             :: {:}\n'.format(args.steps)
    log_str +='num_layers        :: {:}\n'.format(args.nb_layers)
    log_str +='num_classes       :: {:}\n'.format(args.nb_classes)
    log_str +='model_init        :: {:}\n'.format(args.model_init)
    logger.log(log_str, mode='info')

def time_string():
    ISOTIMEFORMAT='%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    return string