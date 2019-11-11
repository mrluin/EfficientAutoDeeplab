import os
import logging
import torch

from run_manager import RunConfig
from nas_manager import ArchSearchRunManager
from utils.common import time_for_file
from utils.common import print_experiment_environment
from utils.common import set_logger
from utils.common import set_manual_seed
from configs.train_search_config import obtain_train_search_args
from models.auto_deeplab import AutoDeepLab


if __name__ == '__main__':
    '''
    # Workspace construct
    # args.path -->| save_path: save ckpts: latest.txt, checkpoint-epoch.pth.tar, checkpoint-best.pth.tar,
    #                                       checkpoint-warmup.pth.tar
    #              | log_path: save logs: net_info.txt, net.config, run.config
    #                                     gradient_search.txt, arch_txt, train_console.txt, valid_test_console.txt
    #              | prediction: save predictions
    #              | learned_net: normal network configs: net.config, run.config, init
    #              | dataset+'_classes_weights.npy  
    '''
    '''
    # Noting: 1.get ride of logging print.
    '''
    args = obtain_train_search_args()
    set_manual_seed(args.random_seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    print_experiment_environment()

    #os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_ids)
    os.makedirs(args.path, exist_ok=True)

    EXP_time = time_for_file()
    # /home/jingweipeng/ljb/Jingbo.TTB/proxy_auto_deeplab/exp_time
    args.path = os.path.join(args.path, args.exp_name, EXP_time)
    # build run configs
    args.lr_scheduler_param = None
    args.optim_params = {
        'momentum': args.momentum,
        'nesterov': args.nesterov
    }
    #print(args.__dict__)
    run_config = RunConfig(
        **args.__dict__
    )

    # debug, adjust run_config
    if args.debug:
        run_config.train_batch_size = None
        run_config.test_batch_size = None
        run_config.valid_size = None
        run_config.workers = None

    # build arch search configs
    if args.arch_optim_type == 'adam':
        args.arch_optim_params = {
            'betas': (args.arch_adam_beta1, args.arch_adam_beta2),
            'eps': args.arch_adam_eps
        }
    else:
        args.arch_optim_params = None
    if args.target_hardware is None:
        args.ref_value = None
    else:
        raise NotImplementedError
    if args.arch_algo == 'grad':
        from nas_manager import GradientArchSearchConfig
        if args.grad_reg_loss_type == 'add#linear':
            args.grad_reg_loss_params = {'lambda': args.grad_reg_loss_lambda}
        elif args.grad_reg_loss_type == 'mul#log':
            args.grad_reg_loss_params = {
                'alpha': args.grad_reg_loss_alpha,
                'beta': args.grad_reg_loss_beta,
            }
        else:
            args.grad_reg_loss_params = None

        arch_search_config = GradientArchSearchConfig(**args.__dict__)
    elif args.arch_algo == 'rl':
        raise NotImplementedError
    else:
        raise NotImplementedError

    #logging.info('Run config:')
    print('Run Configs:')
    for k, v in run_config.config.items():
        #logging.info('\t{}: {}'.format(k, v))
        print('\t{}: {}'.format(k, v))
    #logging.info('Architecture earch config:')
    print('Architecture Search Configs:')
    for k, v in arch_search_config.config.items():
        #logging.info('\t{}: {}'.format(k, v))
        print('\t{}: {}'.format(k, v))

    # TODO: network construct
    args.conv_candidates = [
        '3x3_MBConv3', '3x3_MBConv6',
        '5x5_MBConv3', '5x5_MBConv6',
        '7x7_MBConv3', '7x7_MBConv6',
        'Zero', 'Identity'
    ]
    '''
    # auto_deeplab origin candidates
    args.conv_candidates = [
        '3x3_DWConv', '5x5_DWConv',
        '3x3_DilCon', '5x5_DilConv',
        '3x3_AvgPooling', '3x3_MaxPooling',
        'Zero', 'Identity'
    ]
    '''
    auto_deeplab = AutoDeepLab(
        run_config, arch_search_config, args.conv_candidates
    )
    # arch search run manager
    arch_search_run_manager = ArchSearchRunManager(args.path, auto_deeplab, run_config, arch_search_config)

    # resume warmup checkpoint file
    if args.resume and os.path.exists(args.resume_file):
        try:
            arch_search_run_manager.load_model(ckptfile_path=args.resume_file)
        except Exception:
            warmup_path = os.path.join(arch_search_run_manager.run_manager.save_path,
                                       'checkpoint-warmup.pth.tar')
            if os.path.exists(warmup_path):
                #logging.info('Loading warmup weights')
                print('='*30+'=>\tLoading warmup weights ...')
                arch_search_run_manager.load_model(ckptfile_path=warmup_path)
            else:
                #logging.info('Fail to load models')
                print('='*30+'=>\tFail to load warmup weights ...')

    # warm up
    if arch_search_run_manager.warmup:
        arch_search_run_manager.warm_up(warmup_epochs=args.warmup_epochs)
    # joint training
    arch_search_run_manager.train(fix_net_weights=args.debug)





