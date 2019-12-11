# ===============================
# author : Jingbo Lin
# contact: ljbxd180612@gmail.com
# github : github.com/mrluin
# ===============================

import os
import torch
import glob
import random
import json



from exp.sufficient_update.run_manager import RunConfig
from exp.sufficient_update.gumbel_super_network import GumbelAutoDeepLab
from exp.sufficient_update.nas_manager import ArchSearchConfig, ArchSearchRunManager

from configs.train_search_config import obtain_train_search_args
from utils.common import set_manual_seed, print_experiment_environment, time_for_file, create_exp_dir, configs_resume
from utils.common import save_configs
from utils.flop_benchmark import get_model_infos
from utils.logger import prepare_logger, display_all_families_information
from utils.visdom_utils import visdomer
def main(args):

    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True
    # if resume is True, resume configs and checkpoint from the existing files.
    if args.search_resume:
        # args.resume_file path to ... .../EXP-time
        # resume experiment in a new File, rather than the same file.
        # configs resume
        assert os.path.exists(args.resume_file), 'cannot find the resume file {:}, please re-check'.format(args.resume_file)
        config_file_path = os.path.join(args.resume_file, 'search.config')
        assert os.path.exists(config_file_path), "the path to configs file path {:} is not exists".format(config_file_path)
        f = open(config_file_path, 'r')
        config_dict = json.load(f)
        f.close()
        configs_resume(args, config_dict, 'search')
        # new EXP file initialize
        resume_EXP_time = config_dict['path'].split('/')[-1]
        EXP_time = time_for_file()
        args.path = os.path.join(args.path, args.exp_name, EXP_time+'-resume-{:}'.format(resume_EXP_time))
        os.makedirs(args.path, exist_ok=True)
        create_exp_dir(args.path, scripts_to_save=glob.glob('./*/*.py'))
        save_configs(args.__dict__, args.path, 'search')
        logger = prepare_logger(args)
        logger.log("=> loading configs from the file '{:}' start.".format(args.resume_file), mode='info')
        torch.set_num_threads(args.workers)
        set_manual_seed(args.random_seed)
    else:
        # training initialization
        torch.set_num_threads(args.workers)
        set_manual_seed(args.random_seed)
        EXP_time = time_for_file()
        args.path = os.path.join(args.path, args.exp_name, EXP_time)
        os.makedirs(args.path, exist_ok=True)
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
        # arch_optimizer_config
        if args.arch_optimizer_type == 'adam':
            args.arch_optimizer_params = {
                'betas': (args.arch_adam_beta1, args.arch_adam_beta2),
                'eps': args.arch_adam_eps
            }
        else:
            args.arch_optimizer_params = None
        # related to entropy constraint loss
        if args.reg_loss_type == 'add#linear':
            args.reg_loss_params = {'lambda': args.reg_loss_lambda}
        elif args.reg_loss_type == 'mul#log':
            args.reg_loss_params = {
                'alpha': args.reg_loss_alpha,
                'beta': args.reg_loss_beta
            }
        else:
            args.reg_loss_params = None
        # perform config save, for run_configs and arch_search_configs
        save_configs(args.__dict__, args.path, 'search')
        logger = prepare_logger(args)
        logger.log('=> train-search phase initialization done', mode='info')

    #print(args.optimizer_config)
    run_config = RunConfig( **args.__dict__ )
    arch_search_config = ArchSearchConfig( **args.__dict__ )
    if args.open_vis:
        vis = visdomer(args.port, args.server, args.exp_name, args.compare_phase,
                       args.elements, init_params=None)
    else: vis = None

    # args.bn_momentum and args.bn_eps are not used

    super_network = GumbelAutoDeepLab(
        args.filter_multiplier, args.block_multiplier, args.steps,
        args.nb_classes, args.nb_layers, args.bn_momentum, args.bn_eps, args.search_space, logger, affine=False)

    '''
    from exp.autodeeplab.auto_deeplab import AutoDeeplab
    super_network = AutoDeeplab(args.filter_multiplier, args.block_multiplier, args.steps,
                                args.nb_classes, args.nb_layers, args.search_space, logger, affine=False)
    '''
    '''
    from exp.fixed_network_level.supernetwork import FixedNetwork
    super_network = FixedNetwork(args.filter_multiplier, args.block_multiplier, args.steps, args.nb_classes,
                                 args.nb_layers, args.search_space, logger, affine=False)
    '''
    arch_search_run_manager = ArchSearchRunManager(args.path, super_network, run_config, arch_search_config, logger, vis)
    display_all_families_information(args, 'search', arch_search_run_manager, logger)

    '''
    # get_model_infos, perform inference
    # TODO: modify the way of forward into gdas_forward
    flop, param = get_model_infos(super_network, [1, 3, 512, 512])
    print('||||||| FLOPS & PARAMS |||||||')
    print('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
    '''
    # 1. resume warmup phase
    # 2. resume search phase
    # 3. add last_info log × not last_info, every time, the saved_file name is not consistent, should given resume_file

    # 1. given EXP file time           completed :: resume_file :: ->EXP-time
    # 2. get configs, and load config  completed
    # 3. resume checkpoint             completed


    # TODO: have issue in resume semantics. After resume, it will allocate more GPU memory than the normal one, which will raise OOM in search phase.

    if args.search_resume:
        if os.path.exists(args.resume_file): # resume_file :: path to EXP-time
            logger.log("=> loading checkpoint of the file '{:}' start".format(args.resume_file), mode='info')
            warm_up_checkpoint = os.path.join(args.resume_file, 'checkpoints', 'seed-{:}-warm.pth'.format(args.random_seed))
            search_checkpoint = os.path.join(args.resume_file, 'checkpoints', 'seed-{:}-search.pth'.format(args.random_seed))
            if os.path.exists(search_checkpoint): # resume checkpoint in search phase
                checkpoint = torch.load(search_checkpoint)
                super_network.load_state_dict(checkpoint['state_dict'])
                arch_search_run_manager.run_manager.optimizer.load_state_dict(checkpoint['weight_optimizer'])
                arch_search_run_manager.run_manager.scheduler.load_state_dict(checkpoint['weight_scheduler'])
                arch_search_run_manager.arch_optimizer.load_state_dict(checkpoint['arch_optimizer'])
                arch_search_run_manager.run_manager.monitor_metric = checkpoint['best_monitor'][0]
                arch_search_run_manager.run_manager.best_monitor = checkpoint['best_monitor'][1]
                arch_search_run_manager.warmup = checkpoint['warmup']
                arch_search_run_manager.start_epoch = checkpoint['start_epochs'] # pay attention:: start_epochs and warmup_epoch in nas_manager
                logger.log("=> loading checkpoint of the file '{:}' start with {:}-th epochs in search phase".format(
                    search_checkpoint, checkpoint['start_epochs']), mode='info')
            else: # resume checkpoint in warmup phase
                checkpoint = torch.load(warm_up_checkpoint)
                super_network.load_state_dict(checkpoint['state_dict'])
                arch_search_run_manager.run_manager.optimizer.load_state_dict(checkpoint['weight_optimizer'])
                arch_search_run_manager.run_manager.scheduler.load_state_dict(checkpoint['weight_scheduler'])
                arch_search_run_manager.warmup = checkpoint['warmup']
                arch_search_run_manager.warmup_epoch = checkpoint['warmup_epoch']
                logger.log("=> loading checkpoint of the file '{:}' start with {:}-th epochs in warmup phase".format(warm_up_checkpoint, checkpoint['warmup_epoch']), mode='info')
        else:
            logger.log("=> can not find the file: {:} please re-confirm it\n"
                       "=> start warm-up and search from scratch... ...".format(args.resume_file), mode='info')
    else:
        logger.log("=> start warm-up and search from scratch... ...", mode='info')

    # torch.autograd.set_detect_anomaly(True)
    # warm up phase
    if arch_search_run_manager.warmup:
        arch_search_run_manager.warm_up(warmup_epochs=args.warmup_epochs)
    # train search phase
    arch_search_run_manager.train()

    logger.close()

if __name__ == '__main__':
    args = obtain_train_search_args()
    if args.random_seed is None or args.random_seed < 0: args.random_seed = random.randint(1, 100000)
    main(args)