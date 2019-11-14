import argparse
def obtain_train_search_args():
    '''
    add_argument(--metavar, type, default, action, choices, help)
    :return: args for train_search phrase
    '''
    parser = argparse.ArgumentParser(description='AutoDeeplab train search configs')

    ''' common configs '''
    parser.add_argument('--path', type=str, default='/home/jingweipeng/ljb/Jingbo.TTB/Workspace', help='the path to workspace')
    parser.add_argument('--exp_name', type=str, default='proxy_auto_deeplab')
    parser.add_argument('--gpu_ids', type=int, default=1, help='use single gpu by default')
    parser.add_argument('--random_seed', type=int, default=1)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--resume', default=False, action='store_true', help='checkpoint file if needed')
    parser.add_argument('--resume_file', type=str, default=None)

    parser.add_argument('--workers', type=int, default=4)

    ''' run configs, including network weight training hyperparameters '''

    # debug total_epoch, train_batch_size, test_batch_size

    parser.add_argument('--total_epochs', type=int, default=1)
    # data & dataset
    parser.add_argument('--save_path', type=str, default='/home/jingweipeng/ljb/WHUBuilding', help='root dir of dataset')
    parser.add_argument('--dataset', type=str, default='WHUBuilding', choices=['WHUBuilding'])
    parser.add_argument('--nb_classes', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--valid_size', type=float, default=None, help='validation set split proportion from training set')
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--ori_size', type=int, default=512, help='original image size')
    parser.add_argument('--crop_size', type=int, default=512, help='size of cropped patches')
    # optimization
    parser.add_argument('--init_lr', type=float, default=0.025)
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['poly', 'step', 'cosine'])
    # TODO lr_scheduler_param
    parser.add_argument('--optim_type', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--momentum', type=float, default=0.9) #optim param1
    parser.add_argument('--nesterov', action='store_true') #optim param2
    parser.add_argument('--weight_decay', type=float, default=3e-4)
    # print and save freq
    parser.add_argument('--monitor', type=str, default='max#miou', choices=['max#miou', 'max#fscore'])
    parser.add_argument('--save_ckpt_freq', type=int, default=5)
    parser.add_argument('--validation_freq', type=int, default=1)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--print_save_arch_information', default=False, action='store_true')
    parser.add_argument('--save_normal_net_after_training', default=False, action='store_true')
    parser.add_argument('--print_arch_param_step_freq', type=int, default=10)
    # loss function
    parser.add_argument('--use_unbalanced_weights', default=False, action='store_true')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce'])

    # not sure
    # do not use label_smoothing and no_decay_keys be default
    parser.add_argument('--label_smoothing', type=float, default=0.)
    parser.add_argument('--no_decay_keys', type=str, default=None, choices=[None, 'bn', 'bn#bias'])

    ''' net configs '''
    # TODO cells diversity
    parser.add_argument('--nb_layers', type=int, default=12)
    parser.add_argument('--filter_multiplier', type=int, default=8)
    parser.add_argument('--block_multiplier', type=int, default=5)
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--model_init', type=str, default='he_fout', choices=['he_fin', 'he_fout'])
    parser.add_argument('--init_div_groups', default=False, action='store_true')
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--bn_eps', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0)
    # parser.add_argument('--width_stages', type=str, default='24,40,80,96,192,320')
    # parser.add_argument('--n_cell_stages', type=str, default='4,4,4,4,4,1')
    # parser.add_argument('--stride_stages', type=str, default='2,2,2,1,2,1')

    ''' architecture search config, only using gradient-based algorithm by default '''
    parser.add_argument('--arch_algo', type=str, default='grad', choices=['grad'])
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--arch_init_type', type=str, default='normal', choices=['normal', 'uniform'])
    parser.add_argument('--arch_init_ratio', type=float, default=1e-3)
    parser.add_argument('--arch_optim_type', type=str, default='adam', choices=['sgd', 'adam'])
    parser.add_argument('--arch_lr', type=float, default=3e-3)
    parser.add_argument('--arch_adam_beta1', type=float, default=0) # arch_optim_param1
    parser.add_argument('--arch_adam_beta2', type=float, default=0.999) # arch_optim_param2
    parser.add_argument('--arch_adam_eps', type=float, default=1e-8) # arch_optim_param3
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3)

    # TODO related hardware constraint, None by default
    parser.add_argument('--target_hardware', type=str, default=None, choices=['mobile', 'cpu', 'gpu8', 'flops', None])

    #
    parser.add_argument('--grad_update_arch_param_every', type=int, default=1, help='how often update arch parameters, iterations')
    parser.add_argument('--grad_update_steps', type=int, default=1, help='how many times performing update when updating arch_params')
    parser.add_argument('--grad_binary_mode', type=str, default='full_v2', choices=['full', 'full_v2', 'two'], help='forward and backward mode')
    parser.add_argument('--grad_data_batch', type=int, default=None, help='batch_size of valid set (from training set)')
    parser.add_argument('--grad_reg_loss_type', type=str, default='mul#log', choices=['add#linear', 'mul#log'])
    parser.add_argument('--grad_reg_loss_lambda', type=float, default=1e-1) # reg param
    parser.add_argument('--grad_reg_loss_alpha', type=float, default=0.2)  # reg param
    parser.add_argument('--grad_reg_loss_beta', type=float, default=0.3)  # reg param

    args = parser.parse_args()
    return args

    # TODO not sure configs
    # parser.add_argument('--backbone')
    # related precision parser.add_argument('--opt_level')
    # TODO
    # parser.add_argument('--out_stride')
    # TODO related dataset
    # TODO related parallel training, sync_bn
    # control train or search
    # parser.add_argument('--autodeeplab')

    # cannot resize in segmentation case
    # parser.add_argument('--resize', type=int, default=512)
    # parser.add_argument('--freeze_bn', type=bool, default=False, action='store_true')

    # optimizer parameters
    # parser.add_argument('--min_lr', type=float, default=0.001)


