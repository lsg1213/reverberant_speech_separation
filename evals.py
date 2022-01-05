import random
import os

from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.dsp.normalization import normalize_estimates
import torch
from tqdm import tqdm
from numpy import mean
import soundfile as sf

from utils import get_device, makedir


COMPUTE_METRICS = ["si_sdr", "sdr"]

def evaluate(config, model, dataset, savepath, epoch):
    criterion = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    device = get_device()

    example_num = 5
    save_idx = random.sample(range(len(dataset)), example_num)
    example_path = os.path.join(savepath, 'examples')
    makedir(example_path)
    si_sdris = []

    model.eval()
    with torch.no_grad():
        with tqdm(dataset) as pbar: # 데이터마다 길이가 달라서 dataloader 사용 불가
            for inputs in pbar:
                if 'rir' in dataset.task:
                    mix, sources, idx, cleanmix, clean = inputs
                else:
                    mix, sources, idx = inputs
                mix = mix.to(device)
                sources = sources.to(device)

                logits = model(mix.unsqueeze(0))
                si_sdr, reordered_sources = criterion(logits, sources[None], return_est=True)
                input_si_sdr = criterion(torch.stack([mix, mix], 0).unsqueeze(0), sources[None])
                si_sdri = - (si_sdr - input_si_sdr).tolist() # loss is - si-sdr
                
                est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
                mix_np = mix.cpu().numpy()
                
                est_sources_np_normalized = normalize_estimates(est_sources_np, mix_np)
                si_sdris.append(si_sdri)
                pbar.set_postfix({'si_sdri':mean(si_sdris)})
                
                if idx in save_idx:
                    local_save_dir = os.path.join(example_path, f"ex_{epoch}")
                    makedir(local_save_dir)
                    sf.write(os.path.join(local_save_dir, f"{idx}_mixture.wav"), mix_np, config.sr)
                    # Loop over the sources and estimates
                    for src_idx, src in enumerate(sources):
                        sf.write(os.path.join(local_save_dir, f"s{src_idx}.wav"), src.cpu().numpy(), config.sr)
                    for src_idx, est_src in enumerate(est_sources_np_normalized):
                        sf.write(
                            os.path.join(local_save_dir, f"s{src_idx}_estimate.wav"),
                            est_src,
                            config.sr,
                        )
    return mean(si_sdris)

