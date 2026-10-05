"""Extract frozen, all-layer features from Whisper or MERT for every clip in dataset_A and/or dataset_B.

The model is loaded once and run over each --tasks dataset (default: both). For each task, saves
features/<task>/<model>_<source>.npz (task = year for dataset_A, market for dataset_B) with
    ids         (N,)          sample ids, manifest order
    feats       (N, L, 2*D)   per layer: [mean, std] over time, float16
    lang_probs  (N, n_langs)  Whisper only: language-ID probabilities
    langs       (n_langs,)    Whisper only: language codes for lang_probs columns

Whisper (openai-whisper) runs on the full 30 s; the hidden states are the encoder input plus each
of its blocks (the last one after the final LayerNorm, as Whisper's decoder sees it). Frames are
weighted by vocal activity so silent stretches of the vocal stem do not dilute the pooled stats.

MERT (HuggingFace, trust_remote_code) runs on 5 s windows, which matches its pre-training excerpt
length; frames from all windows of a clip are pooled together.

    pip install -U openai-whisper transformers nnAudio
    python extract_features.py --extractor whisper --source vocals
    python extract_features.py --extractor mert --source no_vocals
    python extract_features.py --extractor mert --source mix
    python extract_features.py --extractor mert --source mix --tasks market   # one dataset only
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

from dataset import TASKS, index_stems

_RESAMPLERS = {}


def load_mono(path, target_sr):
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        if (sr, target_sr) not in _RESAMPLERS:
            _RESAMPLERS[(sr, target_sr)] = T.Resample(sr, target_sr)
        wav = _RESAMPLERS[(sr, target_sr)](wav)
    return wav.mean(dim=0)


def pool(x, w=None):
    """x: (B, T, D), w: (B, T) frame weights or None -> (B, 2*D) [mean, std]."""
    if w is None:
        w = torch.ones(x.shape[:2], device=x.device)
    w = (w / w.sum(dim=1, keepdim=True))[..., None]
    mean = (w * x).sum(dim=1)
    std = ((w * (x - mean[:, None]) ** 2).sum(dim=1) + 1e-8).sqrt()
    return torch.cat([mean, std], dim=1)


def activity_weights(audio, n_frames, threshold_db=-30.0, min_active=10):
    """audio: (B, samples) -> (B, n_frames) 0/1 weights where the frame RMS is within threshold_db of
    the clip's peak. Clips with almost no active frames (instrumental) fall back to uniform weights."""
    hop = audio.shape[1] // n_frames
    rms = audio[:, : n_frames * hop].reshape(audio.shape[0], n_frames, hop).pow(2).mean(dim=2).sqrt()
    db = 20 * torch.log10(rms / (rms.max(dim=1, keepdim=True).values + 1e-8) + 1e-8)
    w = (db > threshold_db).float()
    w[w.sum(dim=1) < min_active] = 1.0
    return w


class WhisperExtractor:
    def __init__(self, name, device):
        import whisper
        self.whisper = whisper
        self.model = whisper.load_model(name, device=device)
        self.device = device
        self.sr = whisper.audio.SAMPLE_RATE

    @torch.no_grad()
    def __call__(self, paths):
        whisper, enc = self.whisper, self.model.encoder
        audio = torch.stack([whisper.pad_or_trim(load_mono(p, self.sr)) for p in paths])
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels).to(self.device)

        # Same as AudioEncoder.forward, but pooling every hidden state on the way
        x = F.gelu(enc.conv1(mel))
        x = F.gelu(enc.conv2(x)).permute(0, 2, 1)
        x = (x + enc.positional_embedding).to(x.dtype)  # (B, 1500, D), 20 ms per frame
        w = activity_weights(audio, x.shape[1]).to(self.device)
        pooled = [pool(x, w)]
        for i, block in enumerate(enc.blocks):
            x = block(x)
            if i == len(enc.blocks) - 1:
                x = enc.ln_post(x)
            pooled.append(pool(x, w))

        # Language ID from the final encoder output (detect_language skips the encoder when given it)
        _, probs = whisper.detect_language(self.model, x)
        langs = list(probs[0].keys())
        lang_probs = np.array([[p[l] for l in langs] for p in probs], dtype=np.float32)
        return torch.stack(pooled, dim=1).cpu(), {"lang_probs": lang_probs, "langs": langs}


class MERTExtractor:
    def __init__(self, name, device, window_seconds=5.0):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True).to(device).eval()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
        self.device = device
        self.sr = self.processor.sampling_rate  # 24 kHz
        self.window = int(window_seconds * self.sr)

        # MERT's remote code ignores output_hidden_states on recent transformers (hidden_states comes
        # back None), so record the hidden states with hooks instead: the input to the first
        # transformer layer, then every layer's output. Same 1 + n_layers states as hidden_states.
        layers_name = next(name for name, m in self.model.named_modules()
                           if name.endswith("encoder.layers") and isinstance(m, torch.nn.ModuleList))
        layers = self.model.get_submodule(layers_name)
        encoder = self.model.get_submodule(layers_name.rsplit(".", 1)[0])
        # Pre-LN ("stable layer norm") encoders apply a final LayerNorm after the last layer, and
        # hidden_states[-1] includes it; post-LN encoders apply theirs before the first layer instead
        stable = getattr(self.model.config, "do_stable_layer_norm", False)
        self._final_norm = encoder.layer_norm if stable else torch.nn.Identity()
        self._states = []
        layers[0].register_forward_pre_hook(lambda _, args: self._states.append(args[0]))
        for layer in layers:
            layer.register_forward_hook(
                lambda _, __, out: self._states.append(out[0] if isinstance(out, tuple) else out))
        self.n_states = len(layers) + 1

    @torch.no_grad()
    def __call__(self, paths):
        out = []
        for p in paths:
            audio = load_mono(p, self.sr)
            n = max(1, audio.shape[0] // self.window)
            windows = audio[: n * self.window].reshape(n, self.window) if audio.shape[0] >= self.window \
                else F.pad(audio, (0, self.window - audio.shape[0]))[None]
            inputs = self.processor(list(windows.numpy()), sampling_rate=self.sr, return_tensors="pt")
            self._states = []
            self.model(inputs["input_values"].to(self.device))
            hidden = self._states[:-1] + [self._final_norm(self._states[-1])]
            assert len(hidden) == self.n_states, f"captured {len(hidden)} hidden states, expected {self.n_states}"
            # Each state (n_windows, T, D) -> all frames of the clip (1, n_windows*T, D)
            out.append(torch.stack([pool(h.reshape(1, -1, h.shape[-1]))[0] for h in hidden]))
        return torch.stack(out).cpu(), {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractor", required=True, choices=["whisper", "mert"])
    parser.add_argument("--source", required=True, choices=["mix", "vocals", "no_vocals"])
    parser.add_argument("--model", default=None, help="Default: large-v3 (whisper) / m-a-p/MERT-v1-330M (mert)")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS),
                        help="year = dataset_A, market = dataset_B")
    parser.add_argument("--stem_dir", default=None,
                        help="Default: each task's stem_dir. Searched recursively, wav or mp3. Only with a single task")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_dir", default="./features", help="Each task's features go to <out_dir>/<task>/")
    args = parser.parse_args()
    if args.stem_dir and len(args.tasks) > 1:
        parser.error("--stem_dir applies to a single task; pass --tasks too, or use each task's default stem_dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.extractor == "whisper":
        args.model = args.model or "large-v3"
        extractor = WhisperExtractor(args.model, device)
    else:
        args.model = args.model or "m-a-p/MERT-v1-330M"
        extractor = MERTExtractor(args.model, device)

    # Resolve every task's files before the (long) extraction, so a missing stem fails fast
    jobs = []
    for task_name in args.tasks:
        task = TASKS[task_name]
        df = pd.read_csv(task["manifest"])
        if args.source == "mix":
            paths = [task["data_dir"] + p for p in df["audio_path"]]
        else:
            stem_dir = args.stem_dir or task["stem_dir"]
            stem_paths = index_stems(stem_dir)
            missing = [sid for sid in df["sample_id"] if (sid, args.source) not in stem_paths]
            if missing:
                raise FileNotFoundError(f"{task_name}: {len(missing)} {args.source} stems missing under {stem_dir}, "
                                        f"e.g. {missing[:3]}")
            paths = [stem_paths[(sid, args.source)] for sid in df["sample_id"]]
        jobs.append((task_name, df["sample_id"].to_numpy().astype(str), paths))

    for task_name, ids, paths in jobs:
        print(f"== {task_name}: {len(paths)} clips")
        out_dir = os.path.join(args.out_dir, task_name)
        out = os.path.join(out_dir, f"{args.model.split('/')[-1]}_{args.source}.npz")
        extract(extractor, ids, paths, out, args.batch_size)


def extract(extractor, ids, paths, out, batch_size):
    feats, extras = [], {}
    for start in range(0, len(paths), batch_size):
        f, extra = extractor(paths[start:start + batch_size])
        feats.append(f.half())
        for k, v in extra.items():
            extras.setdefault(k, []).append(v)
        print(f"[{min(start + batch_size, len(paths))}/{len(paths)}]", flush=True)

    os.makedirs(os.path.dirname(out), exist_ok=True)
    save = {"ids": ids, "feats": torch.cat(feats).numpy()}
    if "lang_probs" in extras:
        save["lang_probs"] = np.concatenate(extras["lang_probs"])
        save["langs"] = np.array(extras["langs"][0])
    np.savez(out, **save)
    print(f"Saved {out}: feats {save['feats'].shape}")


if __name__ == "__main__":
    main()
