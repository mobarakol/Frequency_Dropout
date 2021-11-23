import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--in_dim', type=int, default=3)
    parser.add_argument('--log_name', type=str, default='cbs')
    parser.add_argument('--alg', type=str, default='res', choices=['normal', 'vgg', 'res','resnext', 'wrn'])
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--ssl', action='store_true')
    parser.add_argument('--seed', type=int, default=12)
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--percentage', type=int, default=10)

    # CBS ARGS
    parser.add_argument('--use_cbs', default=False, action='store_true', help='Use CBS or not')
    parser.add_argument('--std', default=1, type=float)
    parser.add_argument('--std_factor', default=0.9, type=float)
    parser.add_argument('--epoch', default=5, type=int)
    parser.add_argument('--kernel_size', default=3, type=int)

    #drop out 
    parser.add_argument('--num_filters', type=int, default=3)
    parser.add_argument('--use_gf', default=False, action='store_true', help='Use FD-GF or not')
    parser.add_argument('--freq_min_all', default=[0.2, 0.2, 0], nargs='+', type=float)
    parser.add_argument('--freq_max_all', default=[1.0, 3.0, 1.0], nargs='+', type=float)
    parser.add_argument('--dropout_p_all', default=[0.4, 0.5, 0.8], nargs='+', type=float)

    args = parser.parse_args()

    args.cuda = True if torch.cuda.is_available() and not args.no_cuda else False

    return args

