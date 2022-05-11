from torchsummary import summary
import torchaudio
import os
import models
import pytorch_model_summary
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# model = torchaudio.models.ConvTasNet(msk_activate='sigmoid').cuda()
model = models.T60_ConvTasNet_v1(None).cuda()
# summary(model, (24000,))
print(pytorch_model_summary.summary(model, [torch.zeros(1, 24000).cuda(), torch.zeros(1,).cuda()], show_input=True))