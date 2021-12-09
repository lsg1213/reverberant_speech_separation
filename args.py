import argparse

from torch.cuda import device_count


def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--datapath', type=str, default='/root/bigdatasets/librimix')
    args.add_argument('--speechnum', type=int, default=2, choices=[2, 3])
    args.add_argument('--batch', type=int, default=16)
    args.add_argument('--sr', type=int, default=8000, choices=[8000, 16000])
    args.add_argument('--tensorboard_path', type=str, default='/tensorboard_log')
    args.add_argument('--task', type=str, default='sep_clean', choices=['sep_clean'])
    return args.parse_args()
    