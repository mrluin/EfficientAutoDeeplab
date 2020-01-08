# ===============================
# author : Jingbo Lin
# contact: ljbxd180612@gmail.com
# github : github.com/mrluin
# ===============================

import os
import torch
import glob
import json

from models.new_gumbel_model import NewGumbelAutoDeeplab
from configs.retrain_config import obtain_retrain_args
from utils.common import set_manual_seed, time_for_file, save_configs, create_exp_dir, configs_resume
#from run_manager import RunConfig, RunManager
from exp.sufficient_update.run_manager import *
from utils.logger import prepare_logger, display_all_families_information
from utils.visdom_utils import visdomer
from models.gumbel_cells import autodeeplab, proxyless, counter, my_search_space


def main(args):
    assert torch.cuda.is_available(), 'CUDA is not available'
    torch.backends.cudnn.enabled       = True
    torch.backends.cudnn.benchmark     = True
    torch.backends.cudnn.deterministic = True

    if args.retrain_resume and args.evaluation == False: # if resume from the last retrain
        config_file_path = os.path.join(args.resume_file, 'retrain.config')
        assert os.path.exists(config_file_path), 'cannot find config_file {:} from the last retrain phase'.format(config_file_path)
        f = open(config_file_path, 'r')
        config_dict = json.load(f)
        f.close()
        configs_resume(args, config_dict, 'retrain') # config resume from the last retrain
        # get EXP_time in last_retrain for flag
        EXP_time_last_retrain = config_dict['path'].split('/')[-1]
        Exp_name_last_retrain = config_dict['path'].split('/')[-2]
        EXP_time = time_for_file()
        args.path = os.path.join(args.path, args.exp_name, EXP_time+'-resume-{:}'.format(Exp_name_last_retrain+'-'+EXP_time_last_retrain))
        torch.set_num_threads(args.workers)
        set_manual_seed(args.random_seed)  # from the last retrain.
        os.makedirs(args.path, exist_ok=True)
        create_exp_dir(args.path, scripts_to_save='../Efficient_AutoDeeplab')
    elif args.retrain_resume == False and args.evaluation:
        config_file_path = os.path.join(args.evaluation_ckpt, 'retrain.config')
        assert os.path.exists(config_file_path), 'cannot find config_file {:} from the best checkpoint'.format(config_file_path)
        f = open(config_file_path, 'r')
        config_dict = json.load(f)
        f.close()
        configs_resume(args, config_dict, 'retrain')
        EXP_time_best_checkpoint = config_dict['path'].split('/')[-1]
        EXP_name_best_checkpoint = config_dict['path'].split('/')[-2]
        EXP_time = time_for_file()
        args.path = os.path.join(args.path, args.exp_name, EXP_time+'-evaluation-{:}'.format(EXP_name_best_checkpoint+'-'+EXP_time_best_checkpoint))
        torch.set_num_threads(args.workers)
        set_manual_seed(args.random_seed)
        os.makedirs(args.path, exist_ok=True)
        create_exp_dir(args.path, scripts_to_save='../Efficient_AutoDeeplab')
    elif args.retrain_resume == False and args.evaluation == False:
        # resume from the searching phrase.
        config_file_path = os.path.join(args.checkpoint_file, 'search.config')
        assert os.path.exists(config_file_path), 'cannot find config_file {:} from the search phase'.format(config_file_path)
        f = open(config_file_path, 'r')
        config_dict = json.load(f)
        f.close()
        args.random_seed = config_dict['random_seed'] # get random_seed
        # get EXP_time in search phase, for flag
        EXP_time_search = config_dict['path'].split('/')[-1]
        EXP_name_search = config_dict['path'].split('/')[-2]
        EXP_time = time_for_file()
        args.path = os.path.join(args.path, args.exp_name, EXP_time + '-resume-{:}'.format(EXP_name_search + '-' + EXP_time_search))
        torch.set_num_threads(args.workers)
        set_manual_seed(args.random_seed)  # from the last retrain phase or search phase.
        os.makedirs(args.path, exist_ok=True)
        create_exp_dir(args.path, scripts_to_save='../Efficient_AutoDeeplab')
    else:
        raise NotImplementedError('invalid mode retrain_resume {:} open_vis {:}'.format(args.retrain_resume, args.open_vis))
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

    # related to entropy constraint loss
    if args.reg_loss_type == 'add#linear':
        args.reg_loss_params = {'lambda1': args.reg_loss_lambda1, 'lambda2':args.reg_loss_lambda2}
    elif args.reg_loss_type == 'add#linear#linearschedule':
        args.reg_loss_params = {
            'lambda1': args.reg_loss_lambda1,
            'lambda2': args.reg_loss_lambda2,
        }
    elif args.reg_loss_type == 'mul#log':
        args.reg_loss_params = {
            'alpha': args.reg_loss_alpha,
            'beta': args.reg_loss_beta}
    else:
        args.reg_loss_params = None
    # save new config, and create logger.
    save_configs(args.__dict__, args.path, 'retrain')
    logger = prepare_logger(args)
    logger.log('=> loading configs {:} from the last retrain phase.'.format(config_file_path) if args.retrain_resume else
               '=> starting retrain from the search phase config {:}.'.format(config_file_path), mode='info')
    # create run_config
    run_config = RunConfig(**args.__dict__)

    # only open_vis in retrain phrase
    if args.open_vis:
        assert args.evaluation, 'invalid mode open_vis {:} and open_test {:}'.format(args.open_vis, args.open_test)
        vis = visdomer(args.port, args.server, args.exp_name, args.compare_phase,
                       args.elements, init_params=None)
    else: vis = None

    if args.evaluation:
        assert os.path.exists(args.evaluation_ckpt), 'cannot find the best checkpoint {:}'.format(args.evaluation_ckpt)
        checkpoint_path = os.path.join(args.evaluation_ckpt, 'checkpoints', 'seed-{:}-retrain-best.pth'.format(args.random_seed))
        checkpoint = torch.load(checkpoint_path)
        actual_path, cell_genotypes = checkpoint['actual_path'], checkpoint['cell_genotypes']
        normal_network = NewGumbelAutoDeeplab(args.nb_layers, args.filter_multiplier, args.block_multiplier,
                                              args.steps, args.nb_classes, actual_path, cell_genotypes,
                                              args.search_space, affine=True)
        evaluation_run_manager = RunManager(args.path, normal_network, logger, run_config, vis, out_log=True)
        normal_network.load_state_dict(checkpoint['state_dict'])
        display_all_families_information(args, 'retrain', evaluation_run_manager, logger)
        logger.log('=> loaded the best checkpoint from {:}, start evaluation'.format(checkpoint_path))

        evaluation_run_manager.validate(is_test=True, use_train_mode=False)

    else:
        # resume from the last retrain
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
            retrain_run_manager.start_epoch = checkpoint['start_epoch'] # has +1
            logger.log('=> loaded checkpoint file {:} from the last retrain phase, starts with {:}-th epoch'.format(checkpoint_path, checkpoint['start_epoch']), mode='info')
        else:
            # from search phrase, load the optimal architecture and perform retrain.
            arch_checkpoint_path = os.path.join(args.checkpoint_file, 'checkpoints', 'seed-{:}-arch-best.pth'.format(args.random_seed))

            # TODO, the best epoch has gotten in advance.
            #checkpoint_path = os.path.join(args.checkpoint_file, 'checkpoints', 'seed-{:}-search-best.pth'.format(args.random_seed))
            #tmp_checkpoint = torch.load(checkpoint_path)
            #best_epoch = tmp_checkpoint['start_epochs'] - 1
            #logger.log('=> best epochs: {:}'.format(best_epoch), mode='info') # get the best_epoch

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

    logger.close()

if __name__ == '__main__':
    args = obtain_retrain_args()
    if args.retrain_resume:
        assert os.path.exists(args.resume_file), 'cannot find resume_file {:} from the last retrain phase'.format(args.resume_file)
    else:
        assert os.path.exists(args.checkpoint_file), 'cannot find checkpoint_file {:} from search phase'.format(args.checkpoint_file)
    main(args)

