from json.encoder import py_encode_basestring
import os
import random

import soundfile as sf
import numpy as np
import pandas as pd
import torch
from glob import glob
from torch.utils.data import Dataset


class LibriMix(Dataset):
    dataset_name = "LibriMix"

    def __init__(
        self, csv_dir, config, task="sep_clean", sample_rate=16000, n_src=2, segment=3, return_id=False
    ):
        self.config = config
        self.mode = csv_dir.split('/')[-1]
        csv_dir = os.path.join('/'.join(csv_dir.split('/')[:-1]), 'metadata/')
        self.csv_dir = csv_dir
        self.task = task
        self.return_id = return_id
        # Get the csv corresponding to the task
        if task == "sep_clean":
            md_file = [f for f in os.listdir(csv_dir) if "clean" in f and self.mode in f and 'mixture' in f and 'rir' not in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        else:
            task = task.replace('_sep_clean', '')
            md_file = [f for f in os.listdir(csv_dir) if "clean" in f and self.mode in f and 'mixture' in f and f'{task}_' in f][0]
            self.csv_path = os.path.join(self.csv_dir, md_file)
        self.segment = segment
        self.sample_rate = sample_rate
        # Open csv file
        self.df = pd.read_csv(self.csv_path)

        # Get rid of the utterances too short
        if self.segment is not None:
            max_len = len(self.df)
            self.seg_len = int(self.segment * self.sample_rate)
            # Ignore the file shorter than the desired_length
            self.df = self.df[self.df["length"] >= self.seg_len]
            print(
                f"Drop {max_len - len(self.df)} utterances from {max_len} "
                f"(shorter than {segment} seconds)"
            )
        else:
            self.seg_len = None
        self.n_src = n_src

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get the row in dataframe
        row = self.df.iloc[idx]
        # Get mixture path
        mixture_path = row["mixture_path"]
        self.mixture_path = mixture_path

        # If there is a seg start point is set randomly
        if self.seg_len is not None:
            start = random.randint(0, row["length"] - self.seg_len)
            stop = start + self.seg_len
        else:
            start = 0
            stop = None
        # If task is enh_both then the source is the clean mixture
        if "sep_clean" in self.task:
            source_path1 = row["source_1_path"]
            source_path2 = row["source_2_path"]
            s1, _ = sf.read(source_path1, dtype="float32", start=start, stop=stop)
            s2, _ = sf.read(source_path2, dtype="float32", start=start, stop=stop)
            clean_sep = torch.from_numpy(np.stack([s1, s2], -1))
            mixture = clean_sep.clone()
        else:
            # Read sources
            source_path = row["label_path"]
            s, _ = sf.read(source_path, dtype="float32", start=start, stop=stop)
            clean_sep = torch.from_numpy(s)
            # Read the mixture
            mixture, _ = sf.read(mixture_path, dtype="float32", start=start, stop=stop)
            # Convert to torch tensor
            mixture = torch.from_numpy(mixture)

        outputs = (mixture, clean_sep)
        if self.return_id:
            id1, id2 = mixture_path.split("/")[-1].split(".")[0].split("_")
            outputs = outputs + ([id1, id2],)
        if vars(self.config).get('t60') is not None:
            outputs = outputs + (torch.tensor(self.df.iloc[idx]['T60']).type(mixture.dtype),)
        return outputs
        
        