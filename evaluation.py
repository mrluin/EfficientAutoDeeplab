# ===============================
# author : Jingbo Lin
# contact: ljbxd180612@gmail.com
# github : github.com/mrluin
# ===============================
import os
import torch
import json
import glob
from configs.evaluation_config import obtain_evaluation_args
from models.new_gumbel_model import NewGumbelAutoDeeplab
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

    config_file_path = os.path.join(args.resume_file, 'retrain.config')
    assert os.path.exists(config_file_path), 'cannot find config file: {:}'.format(config_file_path)
    f = open(config_file_path, 'r')
    config_dict = json.load(f)
    f.close()
    configs_resume(args, config_dict, 'test')
    EXP_time_from_retrain = config_dict['path'].split('/')[-1].split('-resume-')[0]
    EXP_time = time_for_file()
    args.path = os.path.join(args.path, args.exp_name, EXP_time+'-evaluation-{:}'.format(EXP_time_from_retrain))

    torch.set_num_threads(args.workers)
    set_manual_seed(args.random_seed)
    os.makedirs(args.path, exist_ok=True)
    create_exp_dir(args.path, scripts_to_save=glob.glob('./*/*.py'))
    save_configs(args.__dict__, args.path, 'test')
    logger = prepare_logger(args)
    logger.log('=> Loading configs {:} from retrain checkpoint'.format(config_file_path))

    if args.search_space == 'autodeeplab':
        conv_candidates = autodeeplab
    elif args.search_space == 'proxyless':
        conv_candidates = proxyless
    elif args.search_space == 'my_search_space':
        conv_candidates = my_search_space
    else:
        raise ValueError('search space {:} is not supported'.format(args.search_space))

    run_config = RunConfig(**args.__dict__)
    checkpoint_file = os.path.join(args.resume_file, 'checkpoints', 'seed-{:}-retrain-best.pth'.format(args.random_seed))
    assert os.path.join(checkpoint_file), 'cannot find checkpoint file {:}'.format(checkpoint_file)
    logger.log('=> Loading checkpoint from {:} from retrain checkpoint'.format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file)
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
    normal_network = NewGumbelAutoDeeplab(args.nb_layers, args.filter_multiplier, args.block_multiplier, args.steps,
                                          args.nb_classes, args.actual_path, args.cell_genotypes, args.search_space, affine=True)
    evaluation_run_manager = RunManager(args.path, normal_network, logger, run_config)
    normal_network.load_state_dict(checkpoint['state_dict'])
    display_all_families_information(args, 'retrain', evaluation_run_manager, logger)

    # do not need load optimizer state_dicts

    logger.log('=> loaded checkpoint file {:} from the retrain-best'.format(checkpoint_file))

    evaluation_run_manager.validate(is_test=True, use_train_mode=False)
    logger.close()

if __name__ == '__main__':
    args = obtain_evaluation_args()
    assert os.path.exists(args.resume_file), 'cannot find resume_file: {:}'.format(args.resume_file)
    main(args)
