import argparse

from torch.cuda import device_count


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--name', type=str, default='test')
    args.add_argument('--datapath', type=str, default='/root/bigdatasets/librimix')
    args.add_argument('--speechnum', type=int, default=2, choices=[2, 3])
    args.add_argument('--batch', type=int, default=18)
    args.add_argument('--task', type=str, default='')
    args.add_argument('--epoch', type=int, default=200)
    args.add_argument('--max_patience', type=int, default=10)
    args.add_argument('--segment', type=int, default=3)
    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--clip_val', type=float, default=5.)
    args.add_argument('--sr', type=int, default=8000, choices=[8000, 16000])
    args.add_argument('--tensorboard_path', type=str, default='tensorboard_log')
    args.add_argument('--gpus', type=str, default='-1')
    args.add_argument('--resume', action='store_true')
    args.add_argument('--norm', action='store_true')
    return args.parse_args()
    