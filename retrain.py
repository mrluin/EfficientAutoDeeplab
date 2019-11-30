'''
@author: Jingbo Lin
@contact: ljbxd180612@gmail.com
@github: github.com/mrluin
'''
import os
import torch
import glob

from models.new_gumbel_model import NewGumbelAutoDeeplab
from configs.retrain_config import obtain_retrain_args
from utils.common import set_manual_seed, time_for_file, save_configs
from run_manager import RunConfig, RunManager
from utils.logger import prepare_logger, display_all_families_information
from utils.visdom_utils import visdomer


def main(args):
    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.workers)
    set_manual_seed(args.random_seed)
    EXP_time = time_for_file()
    args.path = os.path.join(args.path, args.exp_name, EXP_time)
    # after train_search phase, directory and scripts copy has been created
    # optimizer params
    if args.weight_optimizer_type == 'SGD':
        weight_optimizer_params = {
            'momentum': args.momentum,
            'nesterov': args.nesterov,
            'weight_decay': args.weight_decay,
        }
    elif args.weight_optimizer_type == 'RMSprop':
        weight_optimizer_params = {
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
        }
    else: weight_optimizer_params = None
    # scheduler params
    if args.scheduler == 'cosine':
        scheduler_params = {
            'T_max': args.T_max,
            'eta_min': args.eta_min
        }
    elif args.scheduler == 'multistep':
        scheduler_params = {
            'milestones': args.milestones,
            'gammas': args.gammas
        }
    elif args.scheduler == 'exponential':
        scheduler_params = {'gamma': args.gamma}
    elif args.scheduler == 'linear':
        scheduler_params = {'min_lr': args.min_lr}
    else: scheduler_params = None
    # criterion params
    if args.criterion == 'SmoothSoftmax':
        criterion_params = {'label_smooth': args.label_smoothing}
    else: criterion_params = None

    args.optimizer_config = {
        'optimizer_type': args.weight_optimizer_type,
        'optimizer_params': weight_optimizer_params,
        'scheduler': args.scheduler,
        'scheduler_params': scheduler_params,
        'criterion': args.criterion,
        'criterion_params': criterion_params,
        'init_lr': args.init_lr,
        'epochs': args.epochs,
        'class_num': args.nb_classes,
    }
    args.conv_candidates = [
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
        'Zero', #'Identity'
    ]
    # create run_config
    run_config = RunConfig(**args.__dict__)
    save_configs(run_config.config, None, args.path, 'retrain')
    logger = prepare_logger(args)

    if args.open_test == False:
        if args.open_vis:
            vis = visdomer(args.port, args.server, args.exp_name, args.compare_phase,
                           args.elements, init_params=None)
        else: vis = None
        # get actual_path and cell_genotypes
        logger.log('Reading best arch_checkpoint found in search phase: {:}'.format(args.arch_checkpoint), mode='info')
        checkpoint = torch.load(args.arch_checkpoint)
        actual_path, cell_genotypes = checkpoint['actual_path'], checkpoint['cell_genotypes']
        args.actual_path = actual_path
        args.cell_genotypes = cell_genotypes
        new_genotypes = []
        for _index, genotype in cell_genotypes:
            xlist = []
            for edge_genotype in genotype:
                for (node_str, select_index) in edge_genotype:
                    xlist.append((node_str, args.conv_candidates[select_index]))
            new_genotypes.append((_index, xlist))
        log_str = 'Obtained actual_path and cell_genotypes:\n' \
                  'actual_path: {:}\n' \
                  'genotype:\n'.format(actual_path)
        for _index, genotype in new_genotypes:
            log_str += 'index: {:} arch: {:}\n'.format(_index, genotype)
        logger.log(log_str, mode='info')

        normal_network = NewGumbelAutoDeeplab(args.nb_layers, args.filter_multiplier, args.block_multiplier,
                                              args.steps, args.nb_classes, actual_path, cell_genotypes, args.conv_candidates)
        retrain_run_manager = RunManager(args.path, normal_network, logger, run_config, vis, out_log=True)
        display_all_families_information(args, 'retrain', retrain_run_manager, logger)
        # perform train and validation in train() method
        retrain_run_manager.train()
    else:
        checkpoint = torch.load(args.checkpoint_file)
        assert checkpoint.get('state_dict') is not None, \
            'checkpoint file should includes model state_dict in testing phase. please re-confirm'
        actual_path, cell_genotypes = checkpoint['actual_path'], checkpoint['cell_genotypes']
        normal_network = NewGumbelAutoDeeplab(args.nb_layers, args.filter_multiplier, args.block_multiplier,
                                              args.steps, args.nb_classes, actual_path, cell_genotypes, args.conv_candidates)
        normal_network.load_state_dict(checkpoint['state_dict'])
        test_manager = RunManager(args.path, normal_network, logger, run_config, vis=None, out_log=True)
        display_all_families_information(args, 'retrain', test_manager, logger)
        test_manager.validate(epoch=None, is_test=True, use_train_mode=False)

if __name__ == '__main__':
    args = obtain_retrain_args()
    assert args.checkpoint_file is not None and os.path.exists(args.checkpoint_file), \
        'cannot find checkpoint file {:}'.format(args.checkpoint_file)
    main(args)

