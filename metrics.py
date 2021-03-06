import numpy as np
import torch
from mir_eval.separation import bss_eval_sources


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    if not isinstance(src_ref, np.ndarray):
        src_ref = src_ref.detach().cpu().numpy()
    if not isinstance(src_est, np.ndarray):
        src_est = src_est.detach().cpu().numpy()
    if not isinstance(mix, np.ndarray):
        mix = mix.detach().cpu().numpy()

    src_anchor = mix
    sdr, sir, sar, popt = bss_eval_sources(src_ref, src_est)
    sdr0, sir0, sar0, popt0 = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    # print("SDRi1: {0:.2f}, SDRi2: {1:.2f}".format(sdr[0]-sdr0[0], sdr[1]-sdr0[1]))
    return torch.tensor((avg_SDRi))


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    # print("SISNR base1 {0:.2f} SISNR base2 {1:.2f}, avg {2:.2f}".format(
    #     sisnr1b, sisnr2b, (sisnr1b+sisnr2b)/2))
    # print("SISNRi1: {0:.2f}, SISNRi2: {1:.2f}".format(sisnr1, sisnr2))
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr


def call_SISDR(targets, est_targets, use_as_loss=True):
    EPS = 1e-8
    assert targets.size() == est_targets.size()
    # Step 1. Zero-mean norm
    mean_source = torch.mean(targets, dim=2, keepdim=True)
    mean_estimate = torch.mean(est_targets, dim=2, keepdim=True)
    targets = targets - mean_source
    est_targets = est_targets - mean_estimate
    # Step 2. Pair-wise SI-SDR.
    # [batch, n_src]
    pair_wise_dot = torch.sum(est_targets * targets, dim=2, keepdim=True)
    # [batch, n_src]
    s_target_energy = torch.sum(targets ** 2, dim=2, keepdim=True) + EPS
    # [batch, n_src, time]
    scaled_targets = pair_wise_dot * targets / s_target_energy
    e_noise = est_targets - scaled_targets
    # [batch, n_src]
    pair_wise_sdr = torch.sum(scaled_targets ** 2, dim=2) / (
        torch.sum(e_noise ** 2, dim=2) + EPS
    )
    pair_wise_sdr = 10 * torch.log10(pair_wise_sdr + EPS)
    if use_as_loss:
        return - torch.mean(pair_wise_sdr, dim=-1).mean()
    else:
        return torch.mean(pair_wise_sdr, dim=-1)
    

def get_sisdri(mix, src_ref, src_est):
    sdr = call_SISDR(src_ref, src_est, use_as_loss=False)
    sdr0 = call_SISDR(src_ref, torch.cat([mix, mix], 1), use_as_loss=False)
    return  (sdr - sdr0).mean()
