'''
@author: Jingbo Lin
@contact: ljbxd180612@gmail.com
@github: github.com/mrluin
'''

import os
import torch
import glob

from models.gumbel_super_network import GumbelAutoDeepLab
from run_manager import RunConfig
from nas_manager import ArchSearchConfig, ArchSearchRunManager
from configs.train_search_config import obtain_train_search_args
from utils.common import set_manual_seed, print_experiment_environment, time_for_file, create_exp_dir
from utils.common import save_configs
from utils.flop_benchmark import get_model_infos
def main(args):

    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.workers)
    set_manual_seed(args.random_seed)
    print_experiment_environment()
    os.makedirs(args.path, exist_ok=True)
    EXP_time = time_for_file()
    args.path = os.path.join(args.path, args.exp_name, EXP_time)
    create_exp_dir(args.path, scripts_to_save=glob.glob('./*/*.py'))

    # weight optimizer config, related to network_weight_optimizer, scheduler, and criterion
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
    if args.scheduler == 'cosine':
        # TODO: add additional params in args
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
    if args.criterion == 'SmoothSoftmax':
        criterion_params = {'label_smooth': args.label_smoothing}
    else: criterion_params = None
    # weight_optimizer_config, used in run_manager to get weight_optimizer, scheduler, and criterion.
    args.optimizer_config = {
        'optimizer_type'   : args.weight_optimizer_type,
        'optimizer_params' : weight_optimizer_params,
        'scheduler'        : args.scheduler,
        'scheduler_params' : scheduler_params,
        'criterion'        : args.criterion,
        'criterion_params' : criterion_params,
        'init_lr'          : args.init_lr,
        'warmup_epoch'     : args.warmup_epochs,
        'epochs'           : args.epochs,
        'class_num'        : args.nb_classes,
    }
    # TODO need modification
    args.conv_candidates = [
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
        'Zero', #'Identity'
    ]
    run_config = RunConfig( **args.__dict__ )
    # arch_optimizer_config
    if args.arch_optimizer_type == 'adam':
        args.arch_optimizer_params = {
            'betas': (args.arch_adam_beta1, args.arch_adam_beta2),
            'eps': args.arch_adam_eps
        }
    else: args.arch_optimizer_params = None
    # related to hardware constraint
    # TODO: get rid of
    if args.reg_loss_type == 'add#linear':
        args.reg_loss_params = {'lambda': args.reg_loss_lambda}
    elif args.reg_loss_type == 'mul#log':
        args.reg_loss_params = {
            'alpha': args.reg_loss_alpha,
            'beta': args.reg_loss_beta
        }
    else: args.reg_loss_params = None
    arch_search_config = ArchSearchConfig( **args.__dict__ )
    # perform config save, for run_configs and arch_search_configs
    save_configs(run_config.config, arch_search_config.config, args.path)

    print('Run Configs:')
    for k, v in run_config.config.items():
        print('\t{}: {}'.format(k, v))
    print('Architecture Search Configs:')
    for k, v in arch_search_config.config.items():
        print('\t{}: {}'.format(k, v))
    super_network = GumbelAutoDeepLab(
        args.filter_multiplier, args.block_multiplier, args.steps,
        args.nb_classes, args.nb_layers, args.conv_candidates
    )

    arch_search_run_manager = ArchSearchRunManager(args.path, super_network, run_config, arch_search_config)
    print('||||||| {:10s} |||||||'.format(arch_search_run_manager.run_manager.run_config.dataset))
    print('Train Loader :: Len={:} batch_size={:}'.format(len(arch_search_run_manager.run_manager.run_config.train_loader),
                                                          arch_search_run_manager.run_manager.run_config.train_loader.batch_size))
    print('Valid Loader :: Len={:} batch_size={:}'.format(len(arch_search_run_manager.run_manager.run_config.valid_loader),
                                                          arch_search_run_manager.run_manager.run_config.valid_loader.batch_size))
    print('Test  Loader :: Len={:} batch_size={:}'.format(len(arch_search_run_manager.run_manager.run_config.test_loader),
                                                          arch_search_run_manager.run_manager.run_config.test_loader.batch_size))
    print('||||||| OPTIMIZERS |||||||')
    print('weight_optimizer :: {:}'.format(arch_search_run_manager.run_manager.optimizer))
    print('arch_optimizer   :: {:}'.format(arch_search_run_manager.arch_optimizer))
    print('weight_scheduler :: {:} T_max={:} eta_min={:}'.format(args.scheduler, args.T_max, args.eta_min))
    print('critertion       :: {:}'.format(arch_search_run_manager.run_manager.criterion))

    print('||||||| SearchSpace |||||||')
    print(args.conv_candidates)
    print('||||||| SuperNetwork Config |||||||')
    print('filter_multiplier :: {:}'.format(args.filter_multiplier))
    print('block_multiplier  :: {:}'.format(args.block_multiplier))
    print('steps             :: {:}'.format(args.steps))
    print('num_layers        :: {:}'.format(args.nb_layers))
    print('num_classes       :: {:}'.format(args.nb_classes))
    print('model_init        :: {:}'.format(args.model_init))

    '''
    # get_model_infos, perform inference
    # TODO: modify the way of forward into gdas_forward
    flop, param = get_model_infos(super_network, [1, 3, 512, 512])
    print('||||||| FLOPS & PARAMS |||||||')
    print('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
    '''

    # TODO: perform resume automatically
    # warm up phase
    if arch_search_run_manager.warmup:
        arch_search_run_manager.warm_up(warmup_epochs=args.warmup_epochs)
    # train search phase
    arch_search_run_manager.train()


if __name__ == '__main__':
    args = obtain_train_search_args()
    main(args)