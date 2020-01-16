# author: Jingbo Lin
# contact: ljbxd180612@gmail.com
# github: github.com/mrluin

import argparse

def obtain_retrain_args():
    DEFAULT_PORT = 8097
    DEFAULT_HOSTNAME = 'http://localhost'

    parser = argparse.ArgumentParser(description='GumbelAutoDeeplab-retrain')

    ''' re-train config need to be modified '''
    parser.add_argument('--path', type=str, default='/home/jingweipeng/ljb/Jingbo.TTB/Workspace', help='the path to workspace')
    parser.add_argument('--save_path', type=str, default='/home/jingweipeng/ljb/WHUBuilding', help='root dir of dataset')
    parser.add_argument('--exp_name', type=str, default='GumbelAutoDeeplab-retrain')
    parser.add_argument('--gpu_ids', type=int, default=0)
    # random_seed inherit from the searching experiment.
    parser.add_argument('--open_vis', default=False, action='store_true')
    # resume from the last retrain, resume from searching experiment.
    parser.add_argument('--retrain_resume', default=False, action='store_true', help='resume from retrain phase')
    parser.add_argument('--resume_file', type=str, default=None, help='path to retrain resume file')
    parser.add_argument('--checkpoint_file', type=str, default=None, help='start retrain phase according to search phase')
    parser.add_argument('--evaluation', default=False, action='store_true', help='control testing')
    parser.add_argument('--evaluation_ckpt', default=None, type=str, help='checkpoint used in evaluation phrase')


    parser.add_argument('--epochs', type=int, default=100)

    # to control different settings
    parser.add_argument('--filter_multiplier', type=int, default=32)


    # train optimization
    parser.add_argument('--init_lr', type=float, default=1e-3)  # 5e-4 for Adam
    parser.add_argument('--scheduler', type=str, default='poly', choices=['multistep', 'cosine', 'exponential', 'linear', 'poly'])
    parser.add_argument('--weight_optimizer_type', type=str, default='Adam', choices=['SGD', 'RMSprop', 'Adam'])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--weight_decay', type=float, default=0.0005)


    ''' common configs '''
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--search_space', type=str, default='my_search_space', choices=['autodeeplab', 'proxyless', 'my_search_space'])
    # for visdom

    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--server', type=str, default=DEFAULT_HOSTNAME)
    parser.add_argument('--compare_phase', default=['train', 'valid'])
    parser.add_argument('--elements', default=['loss', 'accuracy', 'miou', 'f1score'])
    # for resume and resume_file

    #parser.add_argument('--resume_from_search', default=False, action='store_true', help='if true, resume from configs and checkpoint from search phase')

    ''' run config '''

    parser.add_argument('--warmup_epochs', type=int, default=None)


    parser.add_argument('--dataset', type=str, default='WHUBuilding', choices=['WHUBuilding'])
    parser.add_argument('--nb_classes', type=int, default=2)

    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--valid_batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)

    parser.add_argument('--valid_size', type=float, default=None)
    parser.add_argument('--ori_size', type=int, default=512)
    parser.add_argument('--crop_size', type=int, default=512)

    # scheduler and scheduler params
    parser.add_argument('--T_max', type=float, default=None) # for cosine
    parser.add_argument('--eta_min', type=float, default=0.001) # for cosine
    parser.add_argument('--milestones', type=float, default=None) # for multisteps
    parser.add_argument('--gammas', type=float, default=None) # for multisteps
    parser.add_argument('--gamma', type=float, default=None) # for exponential
    parser.add_argument('--min_lr', type=float, default=None) # for linear
    # not used
    parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])
    # loss function and its params
    parser.add_argument('--criterion', type=str, default='WeightedSoftmax', choices=['Softmax', 'SmoothSoftmax', 'WeightedSoftmax'])
    parser.add_argument('--use_unbalanced_weights', default=False, action='store_true')
    parser.add_argument('--label_smoothing', type=float, default=0.)
    parser.add_argument('--reg_loss_type', type=str, default='add#linear', choices=['add#linear', 'mul#log'])
    parser.add_argument('--reg_loss_lambda1', type=float, default=1e-1) # reg param not used in retrain
    parser.add_argument('--reg_loss_lambda2', type=float, default=1e-1) # reg param not used in retrain
    parser.add_argument('--reg_loss_alpha', type=float, default=0.2)  # reg param
    parser.add_argument('--reg_loss_beta', type=float, default=0.3)  # reg param
    # monitor and frequency
    parser.add_argument('--monitor', type=str, default='max#miou', choices=['max#miou', 'max#fscore'])
    parser.add_argument('--save_ckpt_freq', type=int, default=5)
    parser.add_argument('--validation_freq', type=int, default=1)
    parser.add_argument('--train_print_freq', type=int, default=30)

    ''' net configs '''
    parser.add_argument('--nb_layers', type=int, default=12)

    parser.add_argument('--block_multiplier', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
    parser.add_argument('--init_div_groups', default=False, action='store_true')
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--bn_eps', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.)

    args = parser.parse_args()
    return args


