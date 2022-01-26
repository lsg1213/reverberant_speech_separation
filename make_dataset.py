import argparse
from multiprocessing import cpu_count
import os
import random
from concurrent.futures import ThreadPoolExecutor
import joblib

import numpy as np
import pandas as pd
from scipy.sparse.construct import rand
import torchaudio
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import makedir


args = argparse.ArgumentParser()
args.add_argument('--datapath', type=str, default='/root/bigdatasets/librimix/Libri2Mix/wav8k/min', help='librimix dataset path')
args.add_argument('--nsrc', type=int, default=2)
args.add_argument('--mic', type=int, default=1)
args.add_argument('--sr', type=int, default=8000)
args.add_argument('--gpus', type=str, default='-1')
args.add_argument('--w', type=str, default='3,10')
args.add_argument('--d', type=str, default='3,10')
args.add_argument('--h', type=str, default='2.5,4')
args.add_argument('--T60', type=str, default='0.1,0.5')
args.add_argument('--snr', type=float, default=5)

config = args.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
import gpuRIR
gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)

def read_source(mixture):
    sources_path_list = []
    for k in mixture.keys():
        if 'source_' in k:
            sources_path_list.append(k)
    sources_list = list(map(lambda x: torchaudio.load(mixture[x])[0], sources_path_list))
    return torch.cat(sources_list)


def get_source_position(room_sz, limit=0.5):
    return list(map(lambda x: random.random() * (x - limit * 2) + limit, room_sz))


def get_RIR(config):
    w = [float(i) for i in config.w.split(',')]
    d = [float(i) for i in config.d.split(',')]
    h = [float(i) for i in config.h.split(',')]
    T60 = [float(i) for i in config.T60.split(',')]
    fs = config.sr
    att_diff = 15.0
    att_max = 60.0
    limit = 0.5 # 벽면에서부터 거리
    distance_limit_between_src_rcv = 0.5 # 0.5 m

    room_sz = [random.random() * (w[1] - w[0]) + w[0], random.random() * (d[1] - d[0]) + d[0], random.random() * (h[1] - h[0]) + h[0]]

    pos_src = np.array([get_source_position(room_sz, limit=limit) for _ in range(config.nsrc)])
    pos_rcv = np.array([get_source_position(room_sz, limit=limit) for _ in range(config.mic)])
    while (((pos_src - pos_rcv)**2).sum(-1) ** 0.5 < distance_limit_between_src_rcv).sum() > 0:
        pos_src = np.array([get_source_position(room_sz, limit=limit) for _ in range(config.nsrc)])
        pos_rcv = np.array([get_source_position(room_sz, limit=limit) for _ in range(config.mic)])

    T60 = random.random() * (T60[1] - T60[0]) + T60[0]

    beta = gpuRIR.beta_SabineEstimation(room_sz, T60) # Reflection coefficients
    Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, T60) # Time to start the diffuse reverberation model [s]
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, T60)	 # Time to stop the simulation [s]
    nb_img = gpuRIR.t2n( Tdiff, room_sz )	# Number of image sources in each dimension

    RIRs = gpuRIR.simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs)

    clean_RIRs = gpuRIR.simulateRIR(room_sz, np.zeros_like(beta), pos_src, pos_rcv, nb_img, Tmax, fs)
    distance = min(*pos_rcv[0], *(np.array(room_sz) - pos_rcv[0])) # the shortest distance between mic and wall
    return RIRs, {'room': room_sz, 'pos_src': pos_src, 'pos_rcv': pos_rcv, 'T60': T60}, distance, clean_RIRs


def main(config):
    modes = ['train-360', 'dev', 'test']
    csvpath = os.path.join(config.datapath, 'metadata')
    
    for mode in modes:
        print(f'{mode} generation...')
        metric_csv = pd.read_csv(os.path.join(csvpath, f'metrics_{mode}_mix_clean.csv'))
        mixture_csv = pd.read_csv(os.path.join(csvpath, f'mixture_{mode}_mix_clean.csv'))
        prefix = 'rir_'
        rir_csv_path = os.path.join(csvpath, f'{prefix}mixture_{mode}_mix_clean.csv')
        rir_metric_csv_path = os.path.join(csvpath, f'{prefix}metrics_{mode}_mix_clean.csv')
        rir_csv = mixture_csv.copy()
        rir_metric_csv = metric_csv.copy()
        rir_configs = []
        distances = []

        def generate(inputs):
            idx, (metric, mixture) = inputs
            sources = read_source(mixture) # [source1(time,), source2(time,)]
            rir_function, rir_config, distance, clean_rir_function = get_RIR(config)
            rir_configs.append(rir_config)
            distances.append(distance)
            
            mixture_save_path = rir_csv.iloc[idx]['mixture_path'].replace('mix_clean', f'{prefix}mix_clean')
            makedir('/'.join(mixture_save_path.split('/')[:-1]))
            rir_csv.at[idx, 'mixture_path'] = mixture_save_path

            label_save_path = rir_csv.iloc[idx]['mixture_path'].replace(f'{prefix}mix_clean', f'{prefix}label_clean')
            makedir('/'.join(label_save_path.split('/')[:-1]))
            print(label_save_path)
            rir_csv.at[idx, 'label_path'] = label_save_path

            # rir cross correlation operation
            sources = sources.cpu()
            rir_function = torch.from_numpy(rir_function.squeeze()).cpu()
            clean_rir_function = torch.from_numpy(clean_rir_function.squeeze()).cpu()
            a_coefficient = F.pad(torch.ones((rir_function.shape[0], 1), device=rir_function.device, dtype=rir_function.dtype), (0, rir_function.shape[-1] - 1))
            rir_sources = torchaudio.functional.lfilter(sources, torch.tensor(a_coefficient), rir_function)
            rir_label = torchaudio.functional.lfilter(sources, torch.tensor(a_coefficient), clean_rir_function)
            
            # rir normalization
            # rir_sources = rir_sources * sources.max(-1, keepdims=True)[0] / rir_sources.max(-1, keepdims=True)[0] / dis

            snr2 = 10 ** (metric['source_2_SNR'] / 20.)
            mixture_wave = torch.cat([rir_sources[:1], rir_sources[1:] * snr2])
            label_wave = torch.cat([rir_label[:1], rir_label[1:] * snr2])
            
            # mixture save
            torchaudio.save(mixture_save_path, mixture_wave.cpu(), config.sr)
            torchaudio.save(label_save_path, label_wave.cpu(), config.sr)
            print(mixture_save_path)
            
        # list(map(generate, enumerate(zip(metric_csv.iloc, mixture_csv.iloc)))) # for debug

        with ThreadPoolExecutor(cpu_count() // 4) as pool:
        # with ThreadPoolExecutor(2) as pool:
            list(pool.map(generate, enumerate(zip(metric_csv.iloc, mixture_csv.iloc))))

        rir_csv = pd.concat([rir_csv, pd.DataFrame(rir_configs)], 1)
        rir_metric_csv['distance'] = distances

        # csv save
        rir_csv.to_csv(rir_csv_path, sep=',', na_rep='NaN')
        rir_metric_csv.to_csv(rir_metric_csv_path, sep=',', na_rep='NaN')


if __name__ == '__main__':
    main(config)