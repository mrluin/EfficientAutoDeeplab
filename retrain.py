'''
@author: Jingbo Lin
@contact: ljbxd180612@gmail.com
@github: github.com/mrluin
'''
import os
import torch
import glob
import json

from models.new_gumbel_model import NewGumbelAutoDeeplab
from configs.retrain_config import obtain_retrain_args
from utils.common import set_manual_seed, time_for_file, save_configs, create_exp_dir, configs_resume
from run_manager import RunConfig, RunManager
from utils.logger import prepare_logger, display_all_families_information
from utils.visdom_utils import visdomer
from models.gumbel_cells import autodeeplab, proxyless, counter, my_search_space


def main(args):
    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = True

    if args.retrain_resume:
        config_file_path = os.path.join(args.resume_file, 'retrain.config')
        assert os.path.exists(config_file_path), 'cannot find config_file {:} from the last retrain phase'.format(config_file_path)
        f = open(config_file_path, 'r')
        config_dict = json.load(f)
        f.close()
        configs_resume(args, config_dict, 'retrain')
        # get EXP_time in last_retrain for flag
        EXP_time_last_retrain = config_dict['path'].split('/')[-1]
        EXP_time = time_for_file()
        args.path = os.path.join(args.path, args.exp_name, EXP_time + '-resume-{:}'.format(EXP_time_last_retrain))
        torch.set_num_threads(args.workers)
        set_manual_seed(args.random_seed)  # from the last retrain phase or search phase.
        os.makedirs(args.path, exist_ok=True)
        create_exp_dir(args.path, scripts_to_save=glob.glob('./*/*.py'))
        save_configs(args.__dict__, args.path, 'retrain')
        logger = prepare_logger(args)
        logger.log('=> loading configs {:} from the last retrain phase.'.format(config_file_path), mode='info')
        if args.search_space == 'autodeeplab':
            conv_candidates = autodeeplab
        elif args.search_space == 'proxyless':
            conv_candidates = proxyless
        elif args.search_space == 'my_search_space':
            conv_candidates = my_search_space
        else: raise ValueError('search space {:} is not supported'.format(args.search_space))
    else:
        # resume partial configs setting and arch_checkpoint from the search phase by default.
        config_file_path = os.path.join(args.checkpoint_file, 'search.config')
        assert os.path.exists(config_file_path), 'cannot find config_file {:} from the search phase'.format(config_file_path)
        f = open(config_file_path, 'r')
        config_dict = json.load(f)
        f.close()
        args.random_seed = config_dict['random_seed']
        # get EXP_time in search phase, for flag
        EXP_time_search = config_dict['path'].split('/')[-1]
        EXP_time = time_for_file()
        args.path = os.path.join(args.path, args.exp_name, EXP_time+'-resume-{:}'.format(EXP_time_search))
        torch.set_num_threads(args.workers)
        set_manual_seed(args.random_seed)  # from the last retrain phase or search phase.
        os.makedirs(args.path, exist_ok=True)
        create_exp_dir(args.path, scripts_to_save=glob.glob('./*/*.py'))
        save_configs(args.__dict__, args.path, 'retrain')
        logger = prepare_logger(args)
        logger.log('=> starting retrain from the search phase config {:}.'.format(config_file_path), mode='info')

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
        else:
            weight_optimizer_params = None
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
        else:
            scheduler_params = None
        # criterion params
        if args.criterion == 'SmoothSoftmax':
            criterion_params = {'label_smooth': args.label_smoothing}
        else:
            criterion_params = None

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
        if args.search_space == 'autodeeplab':
            conv_candidates = autodeeplab
        elif args.search_space == 'proxyless':
            conv_candidates = proxyless
        elif args.search_space == 'counter':
            conv_candidates = counter
        elif args.search_space == 'my_search_space':
            conv_candidates = my_search_space
        else:
            raise ValueError('search_space : {:} is not supported'.format(args.search_space))

    # create run_config
    run_config = RunConfig(**args.__dict__)

    #if args.open_test == False: # retrain and validate
    if args.open_vis: # only open_vis in re-train phase, rather than both re-train and test.
        vis = visdomer(args.port, args.server, args.exp_name, args.compare_phase,
                       args.elements, init_params=None)
    else: vis = None
    if args.retrain_resume:
        logger.log('=> Loading checkpoint from {:} of the last retrain phase'.format(args.resume_file), mode='info')
        # checkpoint_file from the last retrain phase.
        checkpoint_path = os.path.join(args.resume_file, 'checkpoints', 'seed-{:}-retrain.pth'.format(args.random_seed))
        assert os.path.exists(checkpoint_path), 'cannot find retrain checkpoint file {:}'.format(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        actual_path, cell_genotypes = checkpoint['actual_path'], checkpoint['cell_genotypes']
        args.actual_path = actual_path
        args.cell_genotypes = cell_genotypes
        normal_network = NewGumbelAutoDeeplab(args.nb_layers, args.filter_multiplier, args.block_multiplier,
                                              args.steps, args.nb_classes, actual_path, cell_genotypes,
                                              args.search_space, affine=True)
        retrain_run_manager = RunManager(args.path, normal_network, logger, run_config, vis, out_log=True)
        normal_network.load_state_dict(checkpoint['state_dict'])
        display_all_families_information(args, 'retrain', retrain_run_manager, logger)
        retrain_run_manager.optimizer.load_state_dict(checkpoint['weight_optimizer'])
        retrain_run_manager.scheduler.load_state_dict(checkpoint['scheduler'])
        retrain_run_manager.monitor_metric = checkpoint['best_monitor'][0]
        retrain_run_manager.best_monitor = checkpoint['best_monitor'][1]
        retrain_run_manager.start_epoch = checkpoint['start_epoch']
        logger.log('=> loaded checkpoint file {:} from the last retrain phase, starts with {:}-th epoch'.format(checkpoint_path, checkpoint['start_epoch']), mode='info')
    else:
        # todo from the search phase, read the last arch_checkpoint, rather than the best one.
        arch_checkpoint_path = os.path.join(args.checkpoint_file, 'checkpoints', 'seed-{:}-arch.pth'.format(args.random_seed))
        assert os.path.exists(arch_checkpoint_path), 'cannot find arch_checkpoint file {:} from search phase'.format(arch_checkpoint_path)
        checkpoint = torch.load(arch_checkpoint_path)
        actual_path, cell_genotypes = checkpoint['actual_path'], checkpoint['cell_genotypes']
        new_genotypes = []
        for _index, genotype in cell_genotypes:
            xlist = []
            for edge_genotype in genotype:
                for (node_str, select_index) in edge_genotype:
                    xlist.append((node_str, conv_candidates[select_index]))
            new_genotypes.append((_index, xlist))
        log_str = 'Obtained actual_path and cell_genotypes:\n' \
                  'Actual_path: {:}\n' \
                  'Genotype:\n'.format(actual_path)
        for _index, genotype in new_genotypes:
            log_str += 'index: {:} arch: {:}\n'.format(_index, genotype)
        logger.log(log_str, mode='info')
        args.actual_path = actual_path
        args.cell_genotypes = cell_genotypes
        normal_network = NewGumbelAutoDeeplab(args.nb_layers, args.filter_multiplier, args.block_multiplier,
                                              args.steps, args.nb_classes, actual_path, cell_genotypes,
                                              args.search_space, affine=True)
        retrain_run_manager = RunManager(args.path, normal_network, logger, run_config, vis, out_log=True)
        #normal_network.load_state_dict(checkpoint['state_dict'])
        display_all_families_information(args, 'retrain', retrain_run_manager, logger)
        logger.log('=> Construct NewGumbelAutoDeeplab according to the last-arch obtained from search phase', mode='info')
    # perform train and validation in train() method
    retrain_run_manager.train()
    '''
    else: # test phase
        checkpoint_path = os.path.join(args.resume_file, 'checkpoints', 'seed-{:}-retrain-best.pth'.format(args.random_seed))
        assert os.path.exists(checkpoint_path), 'cannot find best checkpoint {:} from the retrain phase'.format(checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        actual_path, cell_genotypes = checkpoint['actual_path'], checkpoint['cell_genotypes']
        normal_network = NewGumbelAutoDeeplab(args.nb_layers, args.filter_multiplier, args.block_multiplier,
                                              args.steps, args.nb_classes, actual_path, cell_genotypes, args.search_space, affine=True)
        normal_network.load_state_dict(checkpoint['state_dict'])
        test_manager = RunManager(args.path, normal_network, logger, run_config, vis=None, out_log=True)
        display_all_families_information(args, 'retrain', test_manager, logger)

        # save testing configs
        save_configs(args.__dict__, args.path, 'test')
        test_manager.validate(epoch=None, is_test=    True, use_train_mode = False)
    '''
    logger.close()

if __name__ == '__main__':
    args = obtain_retrain_args()
    if args.retrain_resume:
        assert os.path.exists(args.resume_file), 'cannot find resume_file {:} from the last retrain phase'.format(args.resume_file)
    else:
        assert os.path.exists(args.checkpoint_file), 'cannot find checkpoint_file {:} from search phase'.format(args.checkpoint_file)
    main(args)

