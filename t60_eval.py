import argparse
import os
from asteroid.losses import pairwise_neg_sisdr
from asteroid.dsp.normalization import normalize_estimates
import torch
from tqdm import tqdm
from numpy import mean
import joblib
import numpy as np

from utils import get_device
from t60_utils import *

def evaluate(config, model, dataset, savepath, epoch, dereverb=False):
    metric = newPITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    device = get_device()

    si_sdris = []
    si_snris = []
    output_scores = []
    input_scores = []

    tmp = {}
    if 'lambdaloss2' in config.name or 'lambda2' in config.name or 'lambdaloss3' in config.name: # 조정 후
        meanstd = joblib.load('mean_std2.joblib')
    elif 'lambdaloss1' in config.name or 'lambda1' in config.name: # 조정 전
        meanstd = joblib.load('mean_std1.joblib')
    for i in meanstd:
        if int(i * 1000) not in tmp:
            tmp[int(i * 1000)] = {}
        for j in meanstd[i]:
            tmp[int(i * 1000)][j] = meanstd[i][j].to(device)
    meanstd = tmp
    calculate_lambda = makelambda(config.name)

    model.eval()
    with torch.no_grad():
        with tqdm(dataset) as pbar: # 데이터마다 길이가 달라서 dataloader 사용 불가
            for inputs in pbar:
                mix, clean, idx, distance, t60 = inputs
                t60 = t60[None].to(device)
                if 'lambda' in config.name:
                    lambda_val = []
                    time = torch.tensor(list(meanstd.keys())).unsqueeze(0).to(device)
                    for i in time.squeeze()[torch.argmin(torch.abs(time - torch.round(t60 * 1000).int().unsqueeze(-1)), -1)].tolist():
                        lambda_val.append(torch.normal(meanstd[i]['mean'], meanstd[i]['std']))
                    rawlambda_val = torch.stack(lambda_val)
                    lambda_val = calculate_lambda(rawlambda_val)
                else:
                    lambda_val = t60

                rev_sep = mix[None].to(device).transpose(1,2)
                clean_sep = clean[None].to(device).transpose(1,2)
                mix = rev_sep.sum(1)
                
                mix_std = mix.std(-1, keepdim=True)
                mix_mean = mix.mean(-1, keepdim=True)
                logits = model((mix - mix_mean) / mix_std, t60=lambda_val)
                logits = logits * mix_std.unsqueeze(1) + mix_mean.unsqueeze(1)

                if config.recursive or config.recursive2:
                    logits = logits.detach()
                    logits.requires_grad_(True)
                    if config.recursive:
                        inputs = [mix, logits.sum(1)]
                    elif config.recursive2:
                        inputs = [logits.sum(1)]
                    for i in range(1, config.iternum):
                        lambda_val = calculate_lambda(rawlambda_val + metric(mix.clone().detach().unsqueeze(1).repeat((1,2,1)), logits))
                        mix = torch.stack(inputs).mean(0)
                        mix_std = mix.std(-1, keepdim=True)
                        mix_mean = mix.mean(-1, keepdim=True)
                        logits = model((mix - mix_mean) / mix_std, t60=lambda_val)
                        logits = logits * mix_std.unsqueeze(1) + mix_mean.unsqueeze(1)
                        
                        if config.recursive:
                            logits = logits.detach()
                            logits.requires_grad_(True)
                            inputs.append(logits.clone().sum(1))
                        elif config.recursive2:
                            inputs = [logits.clone().sum(1)]


                mixcat = rev_sep.sum(1, keepdim=True).repeat((1,2,1))
                output_score = - metric(logits, clean_sep).squeeze()
                input_score = - metric(mixcat, clean_sep).squeeze()
                if isinstance(output_score.tolist(), float):
                    output_score = output_score.unsqueeze(0)
                    input_score = input_score.unsqueeze(0)
                output_scores += output_score.tolist()
                input_scores += input_score.tolist()
                si_snris += (output_score - input_score).tolist()

                progress_bar_dict = {}
                progress_bar_dict['input_SI_SNR'] = np.mean(input_scores)
                progress_bar_dict['out_SI_SNR'] = np.mean(output_scores)
                progress_bar_dict['SI_SNRI'] = np.mean(si_snris)

                pbar.set_postfix(progress_bar_dict)
    return mean(0.), mean(si_snris)


def main(config):
    from t60_train import get_model, makedir, ArgumentError, SummaryWriter, LibriMix

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    gpu_num = torch.cuda.device_count()
    config.batch *= max(gpu_num, 1)   
    
    config.model = 'T60_' + config.model
    name = 't60_' + (config.model if config.model is not '' else 'baseline')
    name += f'_{config.batch}'
    if config.model != '' and config.task == '':
        raise ArgumentError('clean separation model should be baseline model')
    if 'rir' not in config.task:
        config.task = ''
        name += '_clean'
    else:
        name += '_' + config.task
        config.task += '_'
    if config.norm:
        name += '_norm'
    if config.recursive:
        name += f'_recursive_{config.iternum}'
    if config.recursive2:
        name += f'_recursive2_{config.iternum}'
    config.name = name + '_' + config.name if config.name is not '' else ''
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.name + 'hi')
    writer = SummaryWriter(config.tensorboard_path)
    savepath = os.path.join('save', config.name)
    device = get_device()
    makedir(config.tensorboard_path)
    makedir(savepath)

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    gpu_num = torch.cuda.device_count()
    config.batch *= max(gpu_num, 1)
    test_set = LibriMix(
        csv_dir=os.path.join(config.datapath, 'Libri2Mix/wav8k/min/test'),
        config=config,
        task=config.task[:-1],
        sample_rate=config.sr,
        n_src=config.speechnum,
        segment=None,
        return_id=True,
    )

    resume = torch.load(os.path.join(savepath, 'best.pt'))
    model = get_model(config)
    model.load_state_dict(resume['model'])
    model = model.to(device)
    si_sdri, si_snri = evaluate(config, model, test_set, savepath, '')
    writer.add_scalar('test/SI-SDRI', si_sdri, resume['epoch'])
    writer.add_scalar('test/SI-SNRI', si_snri, resume['epoch'])
    

if __name__ == '__main__':
    from args import get_args
    args = argparse.ArgumentParser()
    args.add_argument('--test', action='store_true')
    args.add_argument('--recursive', action='store_true')
    args.add_argument('--recursive2', action='store_true')
    args.add_argument('--t60', type=bool, default=True)
    args.add_argument('--iternum', type=int, default=2)

    main(get_args(args))
    
