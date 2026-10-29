# DeepMIR HW2: Symbolic Pop Piano Generation

Transformer models that generate pop piano music, trained on [POP909](https://github.com/music-x-lab/POP909-Dataset).

| Task | Input | Output | Models |
|---|---|---|---|
| **A. Whole-song generation** | nothing | 32-bar piano piece | REMI Transformer, CP Transformer |
| **B. Prompt continuation** | 8-bar piano prompt | the prompt + 24 generated bars | REMI Transformer, CP Transformer |
| **Bonus. Accompaniment** | melody (+ beat annotations) | piano part for the melody | decoder-only or encoder-decoder Transformer |

The report is in [`report/report.md`](report/report.md).

## Quick start (Colab)

Open [`DeepMIR_hw2.ipynb`](DeepMIR_hw2.ipynb) in Colab with a GPU runtime. Run section 1 (setup) first, then only
the sections you need:

| Section | What it does |
|---|---|
| 1. Setup | clone the repo, install packages, download POP909 |
| 2. Pretrained checkpoints | download the trained models, so you can skip training |
| 3. Training | train REMI, CP and the accompaniment model |
| 4. Task A | generate whole songs |
| 5. Task B | continue the prompts |
| 6. Bonus | generate and score accompaniments |
| 7. Evaluation | metrics, scape plots and report figures |
| 8. Save to Drive | copy outputs or checkpoints to Google Drive |

## Local setup

Tested with Python 3.12.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download POP909 ([Google Drive](https://drive.google.com/file/d/1fw0qjeGzzmVkpz6aA8W3dEXygk9rX01t/view)) and unzip it
here, so the folders are:

```
POP909/
├── training/<song id>/<song id>.mid, beat_midi.txt, ...
└── prompts/
    ├── basic-b/001, 002, 003/    Task B prompts
    └── bonus/<id>/melody.mid, beat_midi.txt, chord_midi.txt
```

Training logs to [Weights & Biases](https://wandb.ai); run `wandb login` first, or add `--no_wandb` to any training
command. Every script picks CUDA, then MPS, then CPU, unless you pass `--device`.

## Usage

### Train

```bash
python train.py     --out_dir checkpoints        # REMI
python train_cp.py  --out_dir checkpoints_cp     # CP, 512-token context (add --seq_len 2048 for a longer one)
python train_accomp.py --arch dec --chord --out_dir checkpoints_accomp_dec_chord   # bonus
```

- Each run saves `best/` (lowest validation loss), `last/` (resumable state) and, for REMI and CP, `epoch_XXX/` after
  every epoch. The tokenizer is saved as `tokenizer.json` next to them.
- To continue after an interruption, run the same script with `--resume <out_dir>/last`. It reuses the original
  settings.
- `python <script> --help` lists every option (epochs, batch size, learning rate, model size, ...).

### Task A: whole-song generation

```bash
python generate.py    --ckpt_dir checkpoints/epoch_005    --tokenizer checkpoints/tokenizer.json    --out_dir generated
python generate_cp.py --ckpt_dir checkpoints_cp/epoch_005 --tokenizer checkpoints_cp/tokenizer.json --out_dir generated_cp
```

20 songs × 32 bars by default (`--n_samples`, `--n_bars`). Sampling: `--temperature` (1.0), `--top_p` (0.95),
`--top_k` (0 = off), `--seed`.

### Task B: prompt continuation

```bash
python continue.py    --checkpoint checkpoints/epoch_005  --tokenizer checkpoints/tokenizer.json --output cont
python generate_cp.py --ckpt_dir checkpoints_cp/epoch_005 --tokenizer checkpoints_cp/tokenizer.json \
                      --prompt_dir POP909/prompts/basic-b --out_dir cont_cp
```

Each prompt in `POP909/prompts/basic-b` is continued from its first 8 bars (`--prompt_bars`) by 24 bars (`--n_bars`).
The output `<prompt>_cont.mid` holds the prompt followed by the continuation.

### Bonus: accompaniment

```bash
python generate_accomp.py --arch dec --ckpt_dir checkpoints_accomp_dec_chord/best --out_dir accomp_dec_chord
python HW2_scripts/eval_bonus.py --prompt_dir POP909/prompts/bonus --gen_midi_dir accomp_dec_chord --output_csv accomp_dec_chord.csv
```

- Each output MIDI holds the prompt's melody (track `MELODY`, unchanged) and the generated `PIANO` track, timed on the
  prompt's `beat_midi.txt`.
- `--sweep` generates and scores every temperature × top-p combination, with a summary in `sweep.csv`.
- For models trained with `--chord`, `chords.csv` lists the chord predicted for each bar.

### Evaluate Tasks A and B

```bash
python eval_midi.py --midi_dir generated_cp --output_csv generated_cp/metrics.csv --scape_plot_dir generated_cp_scape_plots
```

| Metric | What it measures |
|---|---|
| H1, H4 | pitch-class entropy over 1 and 4 bars |
| GS | groove similarity: how alike the onset patterns of different bars are |
| SI_short, SI_mid, SI_long | structureness: how strongly segments of 3–8 s, 8–15 s and ≥ 15 s repeat in the piece |

- `eval_midi.py` converts each MIDI into the TA's token format and computes the same H1/H4/GS as
  `HW2_scripts/eval_basic.py`, so it works on MIDI directly.
- SI is a NumPy re-implementation of MusDr's structureness indicator, computed from MIDI chroma.
- `--scape_plot_dir` saves one self-similarity matrix and fitness scape plot per song. `--no_si` skips SI, which is
  the slowest metric.

### Report figures

```bash
python report/make_real_reference.py --out_dir report/real32   # 100 random 32-bar POP909 windows as the real-data reference
python eval_midi.py --midi_dir report/real32 --output_csv report/real32_metrics.csv --scape_plot_dir report/real32_scape_plots
python report/compare_runs.py      # REMI vs. CP across epochs: report/tables.md, fig1-fig5
python report/context_length.py    # CP with 512 vs. 2048 tokens: report/context_length.md, fig6
```

These read generated folders laid out as `remi/{generated,cont}_00X_temp1_topk0_topp095/`,
`cp_transformer/{generated_cp,cont_cp}_00X_temp1_topk0_topp095/` and `cp_2048/`, for epochs 001, 003 and 005. The
notebook's sections 4–5 write to exactly these folders.

## Method

**Beat alignment.** POP909's MIDI bar lines often don't match the music, so `dataset.load_beat_aligned_score` warps
each song so that every annotated beat in `beat_midi.txt` falls on a quarter note, and the first annotated downbeat
falls on a bar line. Training and prompts use the same alignment.

**Tokenization** ([miditok](https://miditok.readthedocs.io/en/latest/), the same settings for both models):
- 88 pitches (A0–C8), 32 velocity bins, 8 positions per beat, chord and tempo tokens, all tracks in one stream.
- **REMI** ([Huang & Yang, 2020](https://arxiv.org/abs/2002.00212)): one token per event (Bar, Position, Pitch,
  Velocity, Duration, ...).
- **CP / CPWord** ([Hsiao et al., 2021](https://arxiv.org/abs/2101.02402)): one compound token per note or metrical
  event, so a sequence is several times shorter.
- **Augmentation:** extra copies of each training song shifted by ±12 semitones, −4 / +5 velocity, and −0.5 / +1 beat
  of duration.

**Models**
- **REMI:** GPT-2 (12 layers, 8 heads, 512-d) on REMI tokens.
- **CP** (`cp_model.py`): the same GPT-2 backbone with one embedding per token type. It predicts the token family
  (note or metric) first, then every other type conditioned on it. Sampling only allows values that make sense for
  the sampled family.
- **Training windows:**
  - REMI uses non-overlapping 512-token chunks.
  - CP uses random bar-aligned windows (`dataset.RandomCropDataset`), redrawn every epoch, so no part of a song is
    skipped whatever the context length.
- **Accompaniment** (`accomp_data.py`, `accomp_model.py`): its own beat-based vocabulary, trained on random 16-bar
  windows with random transposition.
  - `--arch dec` is a GPT-2 that reads melody and piano interleaved bar by bar, and is trained only on the piano
    tokens.
  - `--arch s2s` is a BART encoder-decoder. Two options make its decoder use the melody: `--bar_emb` and
    `--hist_dropout`.
  - `--chord` makes either model predict a chord token per bar before its notes.

## Files

| File | Purpose |
|---|---|
| `DeepMIR_hw2.ipynb` | Colab notebook running every step |
| `train.py` | train the REMI model; also holds the shared tokenizer settings and data helpers |
| `train_cp.py`, `cp_model.py` | train the CP model / the CP model itself |
| `dataset.py` | beat-aligned MIDI loading, fixed-chunk and random-crop datasets |
| `generate.py`, `continue.py` | REMI: Task A / Task B |
| `generate_cp.py` | CP: Task A, or Task B with `--prompt_dir` |
| `train_accomp.py`, `accomp_data.py`, `accomp_model.py` | bonus: training, tokens and data, BART variant |
| `generate_accomp.py` | bonus: generation, optional sampling sweep scored with `eval_bonus.py` |
| `diag_s2s.py` | bonus: checks whether the encoder-decoder uses the melody (validation loss with the real, a wrong, or no melody) |
| `melody_dependence.py` | bonus: how much generated piano depends on its melody (key, chroma, chord agreement) |
| `eval_midi.py` | Task A / B metrics (H1, H4, GS, SI) and scape plots, straight from MIDI |
| `tokenizer.py` | optional: pre-tokenize POP909 for `train.py --tok_dir` (older pipeline, no beat alignment) |
| `HW2_scripts/` | the TA's scripts: `tokenization.py`, `eval_basic.py`, `eval_bonus.py` |
| `report/` | report, real-data reference, comparison scripts and figures |

## References

- Huang & Yang (2020). [Pop Music Transformer](https://arxiv.org/abs/2002.00212) (REMI).
- Hsiao et al. (2021). [Compound Word Transformer](https://arxiv.org/abs/2101.02402) (CP).
- Wang et al. (2020). [POP909](https://arxiv.org/abs/2008.07142).
- Wu & Yang (2020). [The Jazz Transformer on the Front Line](https://arxiv.org/abs/2008.01307) (MusDr metrics, SI).
- [MidiTok documentation](https://miditok.readthedocs.io/en/latest/).
