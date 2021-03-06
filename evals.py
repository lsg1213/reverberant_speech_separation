import random
import os

from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from asteroid.dsp.normalization import normalize_estimates
import torch
from tqdm import tqdm
from numpy import mean, ndarray
import soundfile as sf

from utils import get_device, makedir


def evaluate(config, model, dataset, savepath, epoch, dereverb=False):
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
                t60 = None
                if len(inputs) == 4:
                    mix, clean, idx, t60 = inputs
                    t60 = t60[None].to(device)
                elif len(inputs) == 3:
                    mix, clean, idx = inputs
                rev_sep = mix[None].to(device).transpose(1,2)
                clean_sep = clean[None].to(device).transpose(1,2)
                if dereverb:
                    mix = rev_sep[:,0]
                    mixcat = rev_sep[:,:1]
                else:
                    mix = rev_sep.sum(1)
                    mixcat = torch.stack([mix, mix], 1)

                if config.norm:
                    mix_std = mix.std(-1, keepdim=True)
                    mix_mean = mix.mean(-1, keepdim=True)
                    mix = (mix - mix_mean) / mix_std

                if config.model in ('', 'tas', 'dprnn') or 'test' in config.model:
                    iternum = vars(config).get('iternum')
                    if iternum is None:
                        logits = model(mix)
                        logits = logits * mix_std.unsqueeze(1) + mix_mean.unsqueeze(1)
                    else:
                        if config.residual:
                            inputs = []
                        for i in range(iternum):
                            if config.residual:
                                inputs.append(mix)
                                mix = torch.stack(inputs).mean(0)
                            mix_std = mix.std(-1, keepdim=True)
                            mix_std = torch.maximum(mix_std, torch.tensor(1e-6, dtype=mix.dtype, device=mix.device))
                            mix_mean = mix.mean(-1, keepdim=True)
                            logits = model((mix - mix_mean) / mix_std)
                            logits = logits * mix_std.unsqueeze(1) + mix_mean.unsqueeze(1)
                            mix = logits.clone().detach().sum(1)

                logits = model(mix, t60=t60, test=True)
                logits = logits * mix_std.unsqueeze(1) + mix_mean.unsqueeze(1)

                si_snr = criterion(logits, clean_sep, return_est=True if 'rir' not in dataset.task else False)
                input_si_snr = criterion(mixcat, clean_sep)
                if 'rir' not in dataset.task:
                    si_snr, reordered_sources = si_snr
                si_snri = - (si_snr - input_si_snr).tolist()
                if 'rir' in dataset.task:
                    si_sdr, reordered_sources = criterion(logits, rev_sep, return_est=True)
                    input_si_sdr = criterion(mixcat, rev_sep)
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
                    for src_idx, src in enumerate(clean[0]):
                        sf.write(os.path.join(local_save_dir, f"s{src_idx}.wav"), src.cpu().numpy(), config.sr)
                    for src_idx, est_src in enumerate(est_sources_np_normalized):
                        sf.write(
                            os.path.join(local_save_dir, f"s{src_idx}_estimate.wav"),
                            est_src,
                            config.sr,
                        )
    return mean(si_sdris), mean(si_snris)

