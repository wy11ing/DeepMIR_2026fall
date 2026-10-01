import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T

import pandas as pd

DECADES = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]
LABEL2IDX = {d: i for i, d in enumerate(DECADES)}
IDX2LABEL = {i: d for i, d in enumerate(DECADES)}

MARKETS = ["US", "UK", "Brazil", "Spain", "Germany", "Italy"]
MARKETS2IDX = {d: i for i, d in enumerate(MARKETS)}
IDX2MARKETS = {i: d for i, d in enumerate(MARKETS)}


class _ClipData(Dataset):
    """Shared audio -> mel pipeline. Subclasses only choose the data folder and label map."""

    def __init__(
        self,
        df,
        data_dir,
        label2idx,
        num_chunks=3,
        target_sr=24000,
        n_ftt=512, f_min=0.0,
        f_max=8000.0,
        n_mels=128,
        train=False,
        random_crop=False,
        crop_seconds=4.0,
        eval_crops=10,
        spec_augment=False,
        freq_mask_param=24,
        time_mask_param=40,
        num_masks=2
    ):
        # To ensure 0-start index
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.label2idx = label2idx
        self.num_chunks = num_chunks
        self.target_sr = target_sr

        # random_crop: train draws `num_chunks` random crops of `crop_seconds` per clip,
        # eval uses `eval_crops` evenly spaced (overlapping) crops. Off = 3 fixed chunks.
        self.train = train
        self.random_crop = random_crop
        self.crop_len = int(crop_seconds * target_sr)
        self.eval_crops = eval_crops

        # SpecAugment (train only): random frequency / time masks on the mel
        self.spec_augment = spec_augment and train
        self.num_masks = num_masks
        self.freq_mask = T.FrequencyMasking(freq_mask_param, iid_masks=True)
        self.time_mask = T.TimeMasking(time_mask_param, iid_masks=True)

        self.mel_transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=self.target_sr,
                n_fft=n_ftt,
                f_min=f_min,
                f_max=f_max,
                hop_length=256,
                n_mels=n_mels,
                power=2.0
            ),
            T.AmplitudeToDB(top_db=80)
        )

    def __len__(self):
        return len(self.df)

    def _crops(self, waveform):
        max_start = waveform.shape[0] - self.crop_len
        if self.train:
            starts = torch.randint(0, max_start + 1, (self.num_chunks,)).tolist()
        else:
            starts = torch.linspace(0, max_start, self.eval_crops).long().tolist()
        return torch.stack([waveform[s:s + self.crop_len] for s in starts])

    def _spec_augment(self, mel):
        # mel: (N, 1, n_mels, T). Fill masks with the spectrogram mean, since dB values are
        # not centred on 0 and a 0 fill would look like a loud band.
        fill = mel.mean().item()
        for _ in range(self.num_masks):
            mel = self.freq_mask(mel, mask_value=fill)
            mel = self.time_mask(mel, mask_value=fill)
        return mel

    def __getitem__(self, index):
        # Loading audio and transforming to mel-sepctrogram
        row = self.df.iloc[index]
        wav_path = self.data_dir + row["audio_path"]

        waveform, sr = torchaudio.load(wav_path)
        if sr != self.target_sr:
            waveform = T.Resample(sr, self.target_sr)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        expected_len = 30*self.target_sr
        if waveform.shape[0] < expected_len:
            waveform = nn.functional.pad(waveform, (0, expected_len - waveform.shape[0]))
        else:
            waveform = waveform[:expected_len]

        # Split into chunks: (num_chunks, samples_per_chunk)
        if self.random_crop:
            chunks = self._crops(waveform)
        else:
            chunks = torch.stack(waveform.chunk(self.num_chunks, dim=0))

        # Transform to mel
        mel_spec = self.mel_transform(chunks)

        # Add a channel dimension
        mel_spec = mel_spec.unsqueeze(1)

        if self.spec_augment:
            mel_spec = self._spec_augment(mel_spec)

        # Map label string to class index; test rows have no label -> -1
        label = self.label2idx.get(row["label"], -1)
        label = torch.tensor(label, dtype=torch.long)

        return mel_spec, label


class MusicYearData(_ClipData):
    """Task 1: release-decade classification (dataset_A)."""

    def __init__(self, df, **kwargs):
        super().__init__(df, data_dir="./dataset_A/", label2idx=LABEL2IDX, **kwargs)


class MusicMarketData(_ClipData):
    """Task 2: release-market classification (dataset_B)."""

    def __init__(self, df, **kwargs):
        super().__init__(df, data_dir="./dataset_B/", label2idx=MARKETS2IDX, **kwargs)


# Everything task-specific in one place, used by train.py and inference.py
TASKS = {
    "year": {
        "dataset": MusicYearData,
        "classes": DECADES,
        "manifest": "./dataset_A/manifest.csv",
        "submission_key": "dataset_A",
    },
    "market": {
        "dataset": MusicMarketData,
        "classes": MARKETS,
        "manifest": "./dataset_B/manifest.csv",
        "submission_key": "dataset_B",
    },
}
