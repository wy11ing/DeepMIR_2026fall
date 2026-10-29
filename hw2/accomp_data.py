"""
Melody -> piano accompaniment data (bonus task), shared by both models in train_accomp.py.

Timing comes from beat_midi.txt, not from the MIDI grid: the bonus prompts carry a placeholder 120 BPM tempo,
and eval_bonus.py cuts bars in *seconds* using the same beat file. So a note's position is measured in
annotated beats (SUBS sub-beats per beat), and bars are the spans between annotated downbeats. Decoding maps
sub-beats back to seconds through the same beats, so generated notes line up with eval_bonus.py's bar grid.

Tokens (one vocabulary for both models):
  bar header:      Bar_i (index inside the window)  Beats_n (beats in this bar)  Tempo_t
  melody note:     Pos_p Pitch_x Dur_d                (Pos only when it changes, as in REMI)
  piano note:      Pos_p Pitch_x Vel_v Dur_d
  decoder-only:    BOS | Bar Beats Tempo MEL <melody notes> ACC <piano notes> | Bar ...   (bar-interleaved)
  seq2seq:         encoder BOS | Bar Beats Tempo <melody notes> | ... EOS
                   decoder        | Bar <piano notes> | ... EOS

With chord=True, each bar's piano part starts with a Chord token that the model predicts from the melody
(decoder-only: ... MEL <melody> Chord ACC <piano>; seq2seq decoder: Bar Chord <piano>). The labels come from
the training songs' chord_midi.txt, simplified like eval_bonus.py: one dominant chord per bar, one of 24
major / minor triads, Chord_N (no chord) or Chord_X (dim / aug / sus). At generation time the chord is
sampled, never read from the prompt (its chord_midi.txt is what CA is scored against).
"""
import random
from pathlib import Path

import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset

SUBS = 4            # sub-beats per annotated beat (16th notes when a beat is a quarter)
MAX_BEATS = 8       # longer bars (a handful in POP909) are cut to this many beats
MAX_BARS = 32       # Bar_i indices inside one window
MAX_DUR = 32        # in sub-beats (8 beats)
PITCH_RANGE = (21, 109)
N_TEMPO = 16
TEMPO_RANGE = (40, 240)

def build_vocab():
    toks = ["PAD", "BOS", "EOS", "MEL", "ACC"]
    toks += [f"Bar_{i}" for i in range(MAX_BARS)]
    toks += [f"Beats_{i}" for i in range(1, MAX_BEATS + 1)]
    toks += [f"Tempo_{i}" for i in range(N_TEMPO)]
    toks += [f"Pos_{i}" for i in range(MAX_BEATS * SUBS)]
    toks += [f"Pitch_{i}" for i in range(*PITCH_RANGE)]
    toks += [f"Vel_{i}" for i in range(32)]  # velocity // 4, the same 32 classes eval_bonus.py uses
    toks += [f"Dur_{i}" for i in range(1, MAX_DUR + 1)]
    # appended last, so the ids above (and checkpoints trained before chords existed) are unchanged
    toks += [f"Chord_{chord_name(c)}" for c in range(N_CHORDS)]
    return {t: i for i, t in enumerate(toks)}

# Chord classes: 0-11 major triad on root r, 12-23 minor triad on root r - 12, then no chord / other
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORD_N, CHORD_X, N_CHORDS = 24, 25, 26
ROOTS = {'C': 0, 'B#': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'Fb': 4,
         'F': 5, 'E#': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
         'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11}

def chord_name(c):
    if c == CHORD_N:
        return "N"
    if c == CHORD_X:
        return "X"
    return f"{NOTE_NAMES[c % 12]}:{'min' if c >= 12 else 'maj'}"

VOCAB = build_vocab()
ID2TOK = {i: t for t, i in VOCAB.items()}
PAD, BOS, EOS, MEL, ACC = (VOCAB[t] for t in ("PAD", "BOS", "EOS", "MEL", "ACC"))
CHORD0 = VOCAB[f"Chord_{chord_name(0)}"]

def token_type(i):
    return ID2TOK[i].split("_")[0]

def parse_chord(label):
    """A POP909 chord label -> chord class, with the same simplification as eval_bonus.parse_pop909_chord."""
    if label == "N":
        return CHORD_N
    parts = label.split(":")
    root = ROOTS.get(parts[0], -1)
    if root == -1:
        return CHORD_X
    quality = (parts[1] if len(parts) > 1 else "maj").split("/")[0].split("(")[0]
    if quality.startswith(("dim", "hdim", "aug")) or "sus" in quality:
        return CHORD_X
    return root + 12 if quality.startswith("min") else root

def chord_token(c, transpose=0):
    if c < 24:
        c = (c % 12 + transpose) % 12 + (12 if c >= 12 else 0)
    return CHORD0 + c

def bar_chords(chord_path, grid):
    """Per-bar chord class: the label covering most of the bar (eval_bonus.dominant_chord), N if none."""
    rows = []
    for line in open(chord_path):
        p = line.split()
        if len(p) >= 3:
            rows.append((float(p[0]), float(p[1]), parse_chord(p[2])))
    chords = []
    for k in range(grid.n_bars):
        t0, t1 = grid.times[grid.down[k]], grid.times[grid.down[k + 1]]
        votes = {}
        for s, e, c in rows:
            ov = min(e, t1) - max(s, t0)
            if ov > 0:
                votes[c] = votes.get(c, 0.0) + ov
        chords.append(max(votes, key=votes.get) if votes else CHORD_N)
    return chords

def tempo_bin(beat_len):
    bpm = np.clip(60.0 / beat_len, *TEMPO_RANGE)
    lo, hi = np.log(TEMPO_RANGE[0]), np.log(TEMPO_RANGE[1])
    return int(round((np.log(bpm) - lo) / (hi - lo) * (N_TEMPO - 1)))

class BeatGrid:
    """Seconds <-> annotated beats, linear between beats and extrapolated with the end beat lengths."""
    def __init__(self, beat_path):
        beats = np.loadtxt(beat_path, ndmin=2)  # [time(sec), strong/weak beat, is_downbeat]
        self.times = beats[:, 0]
        down = np.flatnonzero(beats[:, 2] == 1.0)
        if len(down) < 2:  # same fallback as eval_bonus.py
            down = np.arange(0, len(self.times), 4)
        if down[0] > 0:
            down = np.r_[0, down]
        self.down = down
        self.idx = np.arange(len(self.times), dtype=float)

    @property
    def n_bars(self):
        return len(self.down) - 1

    def to_beat(self, t):
        t = np.asarray(t, dtype=float)
        b = np.interp(t, self.times, self.idx)
        first, last = self.times[1] - self.times[0], self.times[-1] - self.times[-2]
        b = np.where(t > self.times[-1], self.idx[-1] + (t - self.times[-1]) / last, b)
        return np.where(t < self.times[0], (t - self.times[0]) / first, b)

    def to_sec(self, b):
        b = np.asarray(b, dtype=float)
        t = np.interp(b, self.idx, self.times)
        first, last = self.times[1] - self.times[0], self.times[-1] - self.times[-2]
        t = np.where(b > self.idx[-1], self.times[-1] + (b - self.idx[-1]) * last, t)
        return np.where(b < 0, self.times[0] + b * first, t)

    def bar_info(self, k):
        """(first beat index, beats in bar, tempo bin) of bar k."""
        start, end = self.down[k], self.down[k + 1]
        beat_len = float(np.median(np.diff(self.times[start:end + 1])))
        return int(start), int(min(end - start, MAX_BEATS)), tempo_bin(beat_len)

def quantize(notes, grid, with_vel):
    """pretty_midi notes -> per-bar lists of (pos, pitch, [vel,] dur), sorted by (pos, pitch)."""
    bars = [[] for _ in range(grid.n_bars)]
    if not notes:
        return bars
    starts = grid.to_beat([n.start for n in notes])
    ends = grid.to_beat([n.end for n in notes])
    bar_starts = grid.down[:-1]
    for n, s, e in zip(notes, starts, ends):
        sub = int(round(s * SUBS))
        k = int(np.searchsorted(bar_starts * SUBS, sub, side="right")) - 1
        if k < 0 or k >= grid.n_bars:
            continue
        pos = sub - bar_starts[k] * SUBS
        beats = min(grid.down[k + 1] - grid.down[k], MAX_BEATS)
        if pos >= beats * SUBS:
            continue
        dur = int(np.clip(round((e - s) * SUBS), 1, MAX_DUR))
        bars[k].append((pos, n.pitch, n.velocity // 4, dur) if with_vel else (pos, n.pitch, dur))
    for b in bars:
        b.sort(key=lambda x: (x[0], x[1]))
    return bars

def note_tokens(notes, transpose=0):
    out, last_pos = [], None
    for note in notes:
        pos, pitch, *rest = note
        if pos != last_pos:
            out.append(VOCAB[f"Pos_{pos}"])
            last_pos = pos
        out.append(VOCAB[f"Pitch_{pitch + transpose}"])
        if len(rest) == 2:
            out.append(VOCAB[f"Vel_{rest[0]}"])
        out.append(VOCAB[f"Dur_{rest[-1]}"])
    return out

def pick_track(midi, name, fallback_idx=0):
    for inst in midi.instruments:
        if inst.name.strip().upper() == name:
            return inst
    return midi.instruments[fallback_idx]

def load_song(song_dir):
    """A training song as a list of bars: dict(beats, tempo, mel, acc, chord)."""
    song_dir = Path(song_dir)
    grid = BeatGrid(song_dir / "beat_midi.txt")
    midi = pretty_midi.PrettyMIDI(str(next(song_dir.glob("*.mid"))))
    mel = quantize(pick_track(midi, "MELODY").notes, grid, with_vel=False)
    acc = quantize(pick_track(midi, "PIANO").notes, grid, with_vel=True)
    chord_path = song_dir / "chord_midi.txt"
    chords = bar_chords(chord_path, grid) if chord_path.exists() else [CHORD_N] * grid.n_bars
    bars = []
    for k in range(grid.n_bars):
        _, beats, tempo = grid.bar_info(k)
        bars.append(dict(beats=beats, tempo=tempo, mel=mel[k], acc=acc[k], chord=chords[k]))
    # drop the silent intro / outro so windows don't start in empty bars
    used = [k for k, b in enumerate(bars) if b["mel"] or b["acc"]]
    return bars[used[0]:used[-1] + 1] if used else []

def load_prompt(prompt_dir):
    """A bonus prompt: (bars without piano or chords, BeatGrid, melody notes in seconds)."""
    prompt_dir = Path(prompt_dir)
    grid = BeatGrid(prompt_dir / "beat_midi.txt")
    melody = pick_track(pretty_midi.PrettyMIDI(str(prompt_dir / "melody.mid")), "MELODY")
    mel = quantize(melody.notes, grid, with_vel=False)
    bars = []
    for k in range(grid.n_bars):
        _, beats, tempo = grid.bar_info(k)
        bars.append(dict(beats=beats, tempo=tempo, mel=mel[k], acc=[]))
    return bars, grid, melody

def bar_header(i, bar):
    return [VOCAB[f"Bar_{i}"], VOCAB[f"Beats_{bar['beats']}"], VOCAB[f"Tempo_{bar['tempo']}"]]

def interleaved(bars, transpose=0, chord=False):
    """Decoder-only sequence and its labels (loss on [chord and] piano tokens and the next bar's Bar token only)."""
    ids, labels = [BOS], [-100]
    for i, bar in enumerate(bars):
        head = bar_header(i, bar)
        ids += head
        # the Bar token right after a bar's piano notes is the "this bar is done" signal, so it is learned
        labels += [head[0] if i > 0 else -100] + [-100] * (len(head) - 1)
        mel = [MEL] + note_tokens(bar["mel"], transpose)
        ids += mel
        labels += [-100] * len(mel)
        if chord:
            ids.append(chord_token(bar["chord"], transpose))
            labels.append(ids[-1])
        ids.append(ACC)
        labels.append(-100)
        acc = note_tokens(bar["acc"], transpose)
        ids += acc
        labels += acc
    return ids, labels

def encoder_seq(bars, transpose=0):
    ids = [BOS]
    for i, bar in enumerate(bars):
        ids += bar_header(i, bar) + note_tokens(bar["mel"], transpose)
    return ids + [EOS]

def decoder_seq(bars, transpose=0, chord=False):
    ids = []
    for i, bar in enumerate(bars):
        ids += [VOCAB[f"Bar_{i}"]] + ([chord_token(bar["chord"], transpose)] if chord else [])
        ids += note_tokens(bar["acc"], transpose)
    return ids + [EOS]

def decode_piano(ids, grid, bar_offset=0):
    """Piano tokens (Bar_i Pos Pitch Vel Dur ...) -> pretty_midi notes in seconds on the prompt's beat grid."""
    notes, bar, pos, pitch, vel = [], None, None, None, None
    for i in ids:
        typ, _, val = ID2TOK[i].partition("_")
        if typ == "Bar":
            bar, pos = int(val) + bar_offset, None
        elif typ == "Pos":
            pos = int(val)
        elif typ == "Pitch":
            pitch = int(val)
        elif typ == "Vel":
            vel = int(val)
        elif typ == "Dur" and None not in (bar, pos, pitch, vel) and bar < grid.n_bars:
            start = grid.down[bar] + pos / SUBS
            s, e = grid.to_sec([start, start + int(val) / SUBS])
            notes.append(pretty_midi.Note(velocity=min(127, vel * 4 + 2), pitch=pitch, start=float(s), end=float(e)))
    return notes

class AccompDataset(Dataset):
    """
    Random windows of up to n_bars bars, with a random transposition (both tracks) per sample.
    One "epoch" draws windows_per_song windows from every song. chord=True adds the per-bar Chord tokens.
    """
    def __init__(self, songs, arch, n_bars=16, max_len=2048, max_enc_len=768, transpose=(-5, 6),
                 windows_per_song=8, deterministic=False, chord=False):
        self.songs = [s for s in songs if len(s) > 0]
        self.chord = chord
        self.arch, self.n_bars, self.max_len, self.max_enc_len = arch, n_bars, max_len, max_enc_len
        self.transpose, self.windows_per_song, self.deterministic = transpose, windows_per_song, deterministic

    def __len__(self):
        return len(self.songs) * self.windows_per_song

    def __getitem__(self, idx):
        song = self.songs[idx // self.windows_per_song]
        if self.deterministic:  # validation: evenly spaced windows, no transposition
            rng, w = None, idx % self.windows_per_song
            start = int(w * max(0, len(song) - self.n_bars) / max(1, self.windows_per_song - 1))
            shift = 0
        else:
            rng = random
            start = rng.randint(0, max(0, len(song) - self.n_bars))
            pitches = [n[1] for b in song for n in b["mel"] + b["acc"]] or [60]
            lo = max(self.transpose[0], PITCH_RANGE[0] - min(pitches))
            hi = min(self.transpose[1], PITCH_RANGE[1] - 1 - max(pitches))
            shift = rng.randint(lo, hi) if lo <= hi else 0
        bars = song[start:start + self.n_bars]
        if self.arch == "dec":
            ids, labels = interleaved(bars, shift, self.chord)
            return {"input_ids": ids[:self.max_len], "labels": labels[:self.max_len]}
        return {"input_ids": encoder_seq(bars, shift)[:self.max_enc_len],
                "labels": decoder_seq(bars, shift, self.chord)[:self.max_len]}

def collate(batch):
    def pad(seqs, value):
        n = max(len(s) for s in seqs)
        return torch.tensor([s + [value] * (n - len(s)) for s in seqs])
    ids = pad([b["input_ids"] for b in batch], PAD)
    return {"input_ids": ids, "attention_mask": (ids != PAD).long(),
            "labels": pad([b["labels"] for b in batch], -100)}
