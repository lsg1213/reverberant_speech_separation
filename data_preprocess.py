import os
import joblib
import torch

import torchaudio
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from args import get_args
from glob import glob
import pandas as pd
import random
import numpy as np
import gpuRIR
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
import torch.nn.functional as F


def get_source_position(room_sz, limit=0.5):
    return list(map(lambda x: random.random() * (x - limit * 2) + limit, room_sz))


def get_RIR(config, T60):
    w = [float(i) for i in config.w.split(',')]
    d = [float(i) for i in config.d.split(',')]
    h = [float(i) for i in config.h.split(',')]
    fs = config.sr
    att_diff = 15.0
    att_max = 60.0
    limit = 0.5 # 벽면에서부터 거리
    distance_limit_between_src_rcv = 0.5 # 0.5 m

    room_sz = [random.random() * (w[1] - w[0]) + w[0], random.random() * (d[1] - d[0]) + d[0], random.random() * (h[1] - h[0]) + h[0]]

    pos_src = np.array([get_source_position(room_sz, limit=limit) for _ in range(config.nsrc)])
    pos_rcv = np.array([get_source_position(room_sz, limit=limit) for _ in range(config.mic)])
    while ((((pos_src - pos_rcv)**2).sum(-1) ** 0.5 < distance_limit_between_src_rcv).sum() > 0) or \
          ((pos_src[:,-1] > 2).sum() + (pos_src[:,-1] < 1).sum() > 0):
        pos_src = np.array([get_source_position(room_sz, limit=limit) for _ in range(config.nsrc)])
        pos_rcv = np.array([get_source_position(room_sz, limit=limit) for _ in range(config.mic)])

    beta = gpuRIR.beta_SabineEstimation(room_sz, T60) # Reflection coefficients
    Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
    nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension

    RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs)

    clean_RIRs = gpuRIR.simulateRIR(room_sz, np.zeros_like(beta), pos_src, pos_rcv, nb_img, Tmax, fs)
    return RIRs, clean_RIRs


def random_samples(totalnum, number):
    return np.random.choice(np.arange(totalnum), (number,))


def read_sources(s1, s2):
    with ThreadPoolExecutor() as pool:
        s1 = list(pool.map(lambda x: torchaudio.load(x)[0], s1))
        s2 = list(pool.map(lambda x: torchaudio.load(x)[0], s2))
    return list(map(lambda x,y: torch.cat([x, y]), s1, s2))


def lfilter(source, rir, rirclean):
    rir = rir.squeeze()
    rirclean = rirclean.squeeze()
    source = F.pad(source.unsqueeze(0), (rir.shape[-1] - 1, 0))
    rir = torch.cat([rir, rirclean])
    rir_sources = F.conv1d(source.repeat((1,2,1)), rir.unsqueeze(1).flip(-1), groups=rir.shape[0]) * 4 * np.pi
    return rir_sources[:,:2], rir_sources[:,2:]


def getscore(rirsource, label):
    return - PITLossWrapper(pairwise_neg_sisdr)(rirsource.sum(-2, keepdim=True).repeat((1, 2, 1)), label)


def preprocessing(config, T60):
    def _preprocessing(rawsource, snr):
        rir, rir_clean = get_RIR(config, T60)
        rirsource, label = lfilter(rawsource.cuda(), torch.from_numpy(rir).cuda(), torch.from_numpy(rir_clean).cuda())

        snr2 = 10 ** (snr / 20.)
        mixture_wave = torch.cat([rirsource[:1], rirsource[1:] * snr2])
        label_wave = torch.cat([label[:1], label[1:] * snr2])
        return getscore(mixture_wave, label_wave).cpu()
    return _preprocessing


def main(config):
    csv = pd.read_csv('/root/bigdatasets/librimix/Libri2Mix/wav8k/min/metadata/rir_mixture_train-360_mix_clean.csv')
    idx = random_samples(len(csv), 32)
    csv = csv.iloc[idx]
    metriccsv = pd.read_csv('/root/bigdatasets/librimix/Libri2Mix/wav8k/min/metadata/rir_metrics_train-360_mix_clean.csv')
    metriccsv = metriccsv.iloc[idx]

    rawsources = read_sources(csv['source_1_path'], csv['source_2_path'])
    out = {}
    # for T60 in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    for T60 in [0.125]:
        scores = []
        for i in tqdm(range(1000)):
            # with ThreadPoolExecutor() as pool:
            #     score = torch.stack(list(pool.map(preprocessing(config, T60), rawsources, metriccsv['source_2_SNR'])))
            score = torch.stack(list(map(preprocessing(config, T60), (rawsources), metriccsv['source_2_SNR'])))
            scores.append(score)
        scores = torch.stack(scores)
        out[T60] = {'mean': scores.mean(), 'std': scores.std()}
    joblib.dump(out, 'mean_std_tmp.joblib')


if __name__ == '__main__':
    import argparse
    args = argparse.ArgumentParser()

    args.add_argument('--w', type=str, default='3,10')
    args.add_argument('--d', type=str, default='3,10')
    args.add_argument('--h', type=str, default='2.5,4')
    args.add_argument('--nsrc', type=int, default=2)
    args.add_argument('--mic', type=int, default=1)
    main(get_args(args))