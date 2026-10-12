from pathlib import Path

import numpy as np
import torch
from symusic import Note, Score, Tempo, TimeSignature, Track
from torch.utils.data import Dataset

def load_beat_aligned_score(midi_path, tpq=480, tempo_change=0.05):
    """
    Load a POP909 MIDI with its timing taken from the beat_midi.txt next to it.

    POP909's MIDI bar lines don't match the music: in most training songs the annotated downbeats sit
    off the MIDI bar lines (often by a fraction of a beat), and the test prompts carry a placeholder 120 BPM
    tempo. So we ignore the MIDI's own grid and warp every note so the i-th annotated beat lands exactly
    on quarter note i (4/4), with the first annotated downbeat on a bar line. Leading silent bars are dropped.
    A tempo event is written on a beat whenever the local tempo moves more than tempo_change from the last one.
    Falls back to the raw MIDI when there is no beat_midi.txt.
    """
    midi_path = Path(midi_path)
    beat_path = midi_path.parent / "beat_midi.txt"
    if not beat_path.exists():
        return Score(midi_path)

    beats = np.loadtxt(beat_path, ndmin=2)  # [time(sec), strong/weak beat, is_downbeat]
    beat_times = beats[:, 0]
    if len(beat_times) < 2:
        return Score(midi_path)
    downbeats = np.flatnonzero(beats[:, 2] == 1.0)
    first_downbeat = downbeats[0] if len(downbeats) else 0
    beat_lens = np.diff(beat_times)
    beat_len = float(np.median(beat_lens))

    def sec_to_beat(t):
        # linear between annotated beats, extrapolated with the median beat length outside them;
        # beat 0 is the first annotated downbeat
        if t < beat_times[0]:
            b = (t - beat_times[0]) / beat_len
        elif t > beat_times[-1]:
            b = len(beat_times) - 1 + (t - beat_times[-1]) / beat_len
        else:
            b = np.interp(t, beat_times, np.arange(len(beat_times)))
        return b - first_downbeat

    src = Score(midi_path, ttype="second")
    notes = [(tr_idx, sec_to_beat(n.time), sec_to_beat(n.end), n.pitch, n.velocity)
             for tr_idx, tr in enumerate(src.tracks) for n in tr.notes]
    if not notes:
        return Score(midi_path)
    # shift by whole bars so the bar of the first note becomes bar 0 (drops silent intros, keeps pickups)
    # (snap to ticks first, so a note 1e-8 beats before a downbeat doesn't count as being in the bar before)
    first_beat = round(min(n[1] for n in notes) * tpq) / tpq
    shift = -4 * np.floor(first_beat / 4)

    def to_tick(b):
        return max(0, round((b + shift) * tpq))

    score = Score(tpq)
    score.time_signatures.append(TimeSignature(0, 4, 4))
    last_qpm = None
    for i, length in enumerate(beat_lens):
        qpm = 60.0 / length
        tick = to_tick(i - first_downbeat)
        if last_qpm is None or (abs(qpm / last_qpm - 1) > tempo_change and tick > 0):
            if tick == 0 and score.tempos:
                score.tempos[-1].qpm = qpm  # beats before the first note all collapse to tick 0
            else:
                score.tempos.append(Tempo(tick, qpm))
            last_qpm = qpm
    tracks = [Track(tr.name, tr.program, tr.is_drum) for tr in src.tracks]
    for tr_idx, start, end, pitch, velocity in notes:
        start, end = to_tick(start), to_tick(end)
        tracks[tr_idx].notes.append(Note(start, max(1, end - start), pitch, velocity))
    for track in tracks:
        track.notes.sort(key=lambda n: (n.time, n.pitch))
        score.tracks.append(track)
    return score

class ChunkMidiDataset(Dataset):
    """Cuts every token sequence into non-overlapping seq_len-token chunks (like REMI's prepare_data)."""
    def __init__(self, sequences, seq_len=512, bos_token_id=None, eos_token_id=None):
        self.seq_len = seq_len
        self.chunks = []
        for ids in sequences:
            # BOS/EOS mark where a song starts and ends, so the model learns how to begin a piece
            if bos_token_id is not None:
                ids = [bos_token_id] + ids
            if eos_token_id is not None:
                ids = ids + [eos_token_id]
            for i in range(0, len(ids)-self.seq_len+1, self.seq_len):
                self.chunks.append(ids[i:i+self.seq_len])

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.chunks[idx])}
