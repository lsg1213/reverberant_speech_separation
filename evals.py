import random
import os

from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.dsp.normalization import normalize_estimates
import torch
from tqdm import tqdm
from numpy import mean
import soundfile as sf

from utils import get_device, makedir


def evaluate(config, model, dataset, savepath, epoch):
    criterion = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    device = get_device()

    example_num = 5
    if savepath != '':
        save_idx = random.sample(range(len(dataset)), example_num)
        example_path = os.path.join(savepath, 'examples')
        makedir(example_path)
    si_sdris = []
    si_snris = []

    model.eval()
    with torch.no_grad():
        with tqdm(dataset) as pbar: # 데이터마다 길이가 달라서 dataloader 사용 불가
            for inputs in pbar:
                if 'rir' in dataset.task:
                    if config.model == '':
                        mix, sources, idx, _, clean = inputs
                    else:
                        mix, sources, idx, _, clean, distance = inputs
                        distance = torch.from_numpy(distance[None]).to(device)
                    mix, sources, clean = mix.to(device), sources.to(device)[None], clean.to(device)[None]
                else:
                    mix, clean, idx = inputs
                    mix, clean = mix.to(device), clean.to(device)[None]

                if config.norm:
                    mix_std = mix.std(-1, keepdim=True)
                    mix_mean = mix.mean(-1, keepdim=True)
                    mix = (mix - mix_mean) / mix_std
                    mix_std = mix_std.unsqueeze(1)
                    mix_mean = mix_mean.unsqueeze(1)

                if config.model == '':
                    logits = model(mix.unsqueeze(0))
                else:
                    logits = model(mix.unsqueeze(0), distance)

                if config.norm:
                    logits = logits * mix_std + mix_mean

                mixcat = torch.stack([mix, mix], 0).unsqueeze(0)

                si_snr = criterion(logits, clean, return_est=True if 'rir' not in dataset.task else False)
                input_si_snr = criterion(mixcat, clean)
                if 'rir' not in dataset.task:
                    si_snr, reordered_sources = si_snr
                si_snri = - (si_snr - input_si_snr).tolist()
                if 'rir' in dataset.task:
                    si_sdr, reordered_sources = criterion(logits, sources, return_est=True)
                    input_si_sdr = criterion(mixcat, sources)
                    si_sdri = - (si_sdr - input_si_sdr).tolist() # loss is - si-sdr
                else:
                    si_sdri = si_snri
                
                est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
                mix_np = mix.cpu().numpy()
                
                est_sources_np_normalized = normalize_estimates(est_sources_np, mix_np)
                si_sdris.append(si_sdri)
                si_snris.append(si_snri)
                pbar.set_postfix({'si_sdri':mean(si_sdris), 'si_snri':mean(si_snris)})
                
                if savepath != '' and idx in save_idx:
                    local_save_dir = os.path.join(example_path, f"ex_{epoch}")
                    makedir(local_save_dir)
                    sf.write(os.path.join(local_save_dir, f"{idx}_mixture.wav"), mix_np, config.sr)
                    # Loop over the sources and estimates
                    for src_idx, src in enumerate(sources[0]):
                        sf.write(os.path.join(local_save_dir, f"s{src_idx}.wav"), src.cpu().numpy(), config.sr)
                    for src_idx, est_src in enumerate(est_sources_np_normalized):
                        sf.write(
                            os.path.join(local_save_dir, f"s{src_idx}_estimate.wav"),
                            est_src,
                            config.sr,
                        )
    return mean(si_sdris), mean(si_snris)

