# author: Jingbo Lin
# contact: ljbxd180612@gmail.com
# github: github.com/mrluin

import argparse

def obtain_retrain_args():
    DEFAULT_PORT = 8097
    DEFAULT_HOSTNAME = 'http://localhost'

    parser = argparse.ArgumentParser(description='GumbelAutoDeeplab-retrain')

    ''' common configs '''
    parser.add_argument('--path', type=str, default='/home/jingweipeng/ljb/Jingbo.TTB/Workspace', help='the path to workspace')
    parser.add_argument('--exp_name', type=str, default='GumbelAutoDeeplab-retrain')
    parser.add_argument('--gpu_ids', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--workers', type=int, default=8)
    # for visdom
    parser.add_argument('--open_vis', default=False, action='store_true')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    parser.add_argument('--server', type=str, default=DEFAULT_HOSTNAME)
    parser.add_argument('--compare_phase', default=['train', 'search'])
    parser.add_argument('--elements', default=['loss', 'accuracy', 'miou', 'f1score'])
    # for resume and resume_file
    parser.add_argument('--checkpoint_file', type=str, default=None, help='arch_checkpoint in retrain phase, checkpoint in testing phase')
    ''' run config '''
    parser.add_argument('--save_path', type=str, default='/home/jingweipeng/ljb/WHUBuilding', help='root dir of dataset')
    parser.add_argument('--dataset', typpe=str, default='WHUBuilding', choices=['WHUBuilding'])
    parser.add_argument('--nb_classes', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--valid_batch_size', type=int, default=8)
    parser.add_argument('--ori_size', type=int, default=512)
    parser.add_argument('--crop_size', type=int, default=512)
    # train optimization
    parser.add_argument('--init_lr', type=float, default=0.025)
    # scheduler and scheduler params
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['multistep', 'cosine', 'exponential', 'linear'])
    parser.add_argument('--T_max', type=float, default=None) # for cosine
    parser.add_argument('--eta_min', type=float, default=0.001) # for cosine
    parser.add_argument('--milestones', type=float, default=None) # for multisteps
    parser.add_argument('--gammas', type=float, default=None) # for multisteps
    parser.add_argument('--gamma', type=float, default=None) # for exponential
    parser.add_argument('--min_lr', type=float, default=None) # for linear
    # optimizer and optimizer params
    parser.add_argument('--weight_optimizer_type', type=str, default='SGD', choices=['SGD','RMSprop'])
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # not used
    parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])
    # loss function and its params
    parser.add_argument('--criterion', type=str, default='Softmax', choices=['Softmax', 'SmoothSoftmax', 'WeightedSoftmax'])
    parser.add_argument('--use_unbalanced_weights', default=False, action='store_true')
    parser.add_argument('--label_smoothing', type=float, default=0.)
    # monitor and frequency
    parser.add_argument('--monitor', type=str, default='max#miou', choices=['max#miou', 'max#fscore'])
    parser.add_argument('--save_ckpt_freq', type=int, default=5)
    parser.add_argument('--validation_freq', type=int, default=1)
    parser.add_argument('--train_print_freq', type=int, default=10)

    ''' net configs '''
    parser.add_argument('--nb_layers', type=int, default=12)
    parser.add_argument('--filter_multiplier', type=int, default=32)
    parser.add_argument('--block_multiplier', type=int, default=1)
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
    parser.add_argument('--init_div_groups', default=False, action='store_true')
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--bn_eps', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.)

    args = parser.parse_args()
    return args


