import os

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


def index_stems(stem_dir):
    """Map (sample_id, stem) -> file for every <sample_id>/<stem>.{wav,mp3} under stem_dir.

    Works with separate.py's layout (<stem_dir>/<id>/vocals.wav) and with the demucs CLI's
    (<stem_dir>/htdemucs/<id>/vocals.mp3), so the stems can live at any depth.
    """
    paths = {}
    for root, _, files in os.walk(stem_dir):
        for f in files:
            stem, ext = os.path.splitext(f)
            key = (os.path.basename(root), stem)
            # Prefer WAV if a clip has both
            if ext == ".wav" or (ext == ".mp3" and key not in paths):
                paths[key] = os.path.join(root, f)
    return paths


class _ClipData(Dataset):
    """Shared audio -> mel pipeline. Subclasses only choose the data folder and label map."""

    def __init__(
        self,
        df,
        data_dir,
        label2idx,
        stem_dir=None,
        stems=("mix",),
        activity_crop=False,
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

        # Each stem becomes one input channel. "mix" is the original clip; anything else is read
        # from <stem_dir>/.../<sample_id>/<stem>.{wav,mp3} (separate.py or the demucs CLI).
        self.stems = list(stems)
        self.stem_paths = {}
        if any(stem != "mix" for stem in self.stems):
            self.stem_paths = index_stems(stem_dir)
            missing = [(sid, stem) for sid in self.df["sample_id"] for stem in self.stems
                       if stem != "mix" and (sid, stem) not in self.stem_paths]
            if missing:
                raise FileNotFoundError(f"{len(missing)} stems missing under {stem_dir}, e.g. {missing[:3]}")
        self._resamplers = {}
        # activity_crop: prefer crops where the vocal stem is active, so random crops do not land
        # in silent stretches (intros, solos, breaks) and teach the model on empty input.
        self.activity_crop = activity_crop and "vocals" in self.stems

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

    def _crop_activity(self, vocals, starts, hop=2400):
        # Fraction of 0.1 s frames inside each crop whose vocal RMS is within 30 dB of the clip's peak
        rms = vocals[: vocals.shape[0] // hop * hop].view(-1, hop).pow(2).mean(dim=1).sqrt()
        active = (20 * torch.log10(rms / (rms.max() + 1e-8) + 1e-8) > -30).float()
        frames = self.crop_len // hop
        return torch.stack([active[s // hop: s // hop + frames].mean() for s in starts.tolist()])

    def _crops(self, waveform):
        # waveform: (S, L) -> (N, S, crop_len)
        max_start = waveform.shape[-1] - self.crop_len
        if self.train:
            num_candidates = self.num_chunks * 4 if self.activity_crop else self.num_chunks
            starts = torch.randint(0, max_start + 1, (num_candidates,))
        else:
            num_candidates = self.eval_crops * 3 if self.activity_crop else self.eval_crops
            starts = torch.linspace(0, max_start, num_candidates).long()

        if self.activity_crop:
            activity = self._crop_activity(waveform[self.stems.index("vocals")], starts)
            num_keep = self.num_chunks if self.train else self.eval_crops
            if self.train:
                # Sample proportionally to activity (floor keeps instrumental clips usable)
                keep = torch.multinomial(activity + 0.05, num_keep, replacement=False)
            else:
                # Deterministic: most active crops, kept in time order
                keep = activity.topk(num_keep).indices.sort().values
            starts = starts[keep]

        return torch.stack([waveform[:, s:s + self.crop_len] for s in starts.tolist()])

    def _load(self, path):
        waveform, sr = torchaudio.load(path)
        if sr != self.target_sr:
            # Demucs MP3s are 44.1 kHz; cache the resampler instead of rebuilding its kernel per clip
            if sr not in self._resamplers:
                self._resamplers[sr] = T.Resample(sr, self.target_sr)
            waveform = self._resamplers[sr](waveform)
        return waveform.mean(dim=0)  # mono

    def _spec_augment(self, mel):
        # mel: (N, S, n_mels, T). Fill masks with the spectrogram mean, since dB values are
        # not centred on 0 and a 0 fill would look like a loud band.
        fill = mel.mean().item()
        for _ in range(self.num_masks):
            mel = self.freq_mask(mel, mask_value=fill)
            mel = self.time_mask(mel, mask_value=fill)
        return mel

    def __getitem__(self, index):
        # Loading audio and transforming to mel-sepctrogram
        row = self.df.iloc[index]

        expected_len = 30*self.target_sr
        waveforms = []
        for stem in self.stems:
            if stem == "mix":
                path = self.data_dir + row["audio_path"]
            else:
                path = self.stem_paths[(row["sample_id"], stem)]
            waveform = self._load(path)
            if waveform.shape[0] < expected_len:
                waveform = nn.functional.pad(waveform, (0, expected_len - waveform.shape[0]))
            waveforms.append(waveform[:expected_len])
        waveform = torch.stack(waveforms)  # (S, L)

        # Split into chunks: (num_chunks, S, samples_per_chunk)
        if self.random_crop:
            chunks = self._crops(waveform)
        else:
            chunks = torch.stack(waveform.chunk(self.num_chunks, dim=-1))

        # Transform to mel: (num_chunks, S, n_mels, T); stems act as input channels
        mel_spec = self.mel_transform(chunks)

        if self.spec_augment:
            mel_spec = self._spec_augment(mel_spec)

        # Map label string to class index; test rows have no label -> -1
        label = self.label2idx.get(row["label"], -1)
        label = torch.tensor(label, dtype=torch.long)

        return mel_spec, label


class MusicYearData(_ClipData):
    """Task 1: release-decade classification (dataset_A)."""

    def __init__(self, df, **kwargs):
        kwargs.setdefault("stem_dir", TASKS["year"]["stem_dir"])
        super().__init__(df, data_dir="./dataset_A/", label2idx=LABEL2IDX, **kwargs)


class MusicMarketData(_ClipData):
    """Task 2: release-market classification (dataset_B)."""

    def __init__(self, df, **kwargs):
        kwargs.setdefault("stem_dir", TASKS["market"]["stem_dir"])
        super().__init__(df, data_dir="./dataset_B/", label2idx=MARKETS2IDX, **kwargs)


# Everything task-specific in one place, used by train.py and inference.py
TASKS = {
    "year": {
        "dataset": MusicYearData,
        "classes": DECADES,
        "manifest": "./dataset_A/manifest.csv",
        "data_dir": "./dataset_A/",
        "stem_dir": "./separated_A",
        "submission_key": "dataset_A",
    },
    "market": {
        "dataset": MusicMarketData,
        "classes": MARKETS,
        "manifest": "./dataset_B/manifest.csv",
        "data_dir": "./dataset_B/",
        "stem_dir": "./separated_B",
        "submission_key": "dataset_B",
    },
}
