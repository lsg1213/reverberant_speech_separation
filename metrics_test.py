from metrics import *
import torchaudio
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class test:
    def __init__(self) -> None:
        self.mix, self.sr = torchaudio.load('/root/bigdatasets/librimix/Libri2Mix/wav8k/max/dev/mix_clean/84-121123-0001_5895-34629-0002.wav')
        self.s1 = torchaudio.load('/root/bigdatasets/librimix/Libri2Mix/wav8k/max/dev/s1/84-121123-0001_5895-34629-0002.wav')[0]
        self.s2 = torchaudio.load('/root/bigdatasets/librimix/Libri2Mix/wav8k/max/dev/s2/84-121123-0001_5895-34629-0002.wav')[0]
        self.call_SDRI_test()

    def call_SDRI_test(self):
        import pdb; pdb.set_trace()
        cal_SDRi(self.mix, torch.cat([self.s1, self.s2], 0), self.mix)
        call_SISDR


if __name__ == '__main__':
    test()