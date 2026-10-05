# DeepMIR HW1: Music Classification

- **Task A** (`dataset_A`): predict a song's release decade (1960s to 2010s) from a 30 s clip.
- **Task B** (`dataset_B`): predict a song's release market (US, UK, Brazil, Spain, Germany, Italy).

Each test clip gets three ranked guesses. The score is `top1 + 0.5 * top3`.

[Slides](https://docs.google.com/presentation/d/1bfitFYh_Gg0QQ3KG8BOB6nq3iUeCORtkSOGmwRmXN_A/edit?slide=id.p#slide=id.p) · [Dataset](https://drive.google.com/drive/folders/1C8RymiLbr-EGmkxh2Ap5TIybnJYqNsb4?usp=drive_link)

## Reproducing the submission

Open **[`DeepMIR_hw1.ipynb`](DeepMIR_hw1.ipynb)** (in Colab or locally) and run all cells. The notebook:

1. clones this repo and installs `requirements.txt`
2. downloads `dataset_A`, `dataset_B` and the pre-extracted Task B features
3. downloads the two checkpoints into `checkpoints/`
4. runs Task A inference: `python inference.py --ckpt checkpoints/taskA_shortchunk_cnn.pt --output predictions_A.json`
5. runs Task B inference: `python inference_features.py --ckpt checkpoints/taskB_dense_head.pt --output predictions_B.json`
6. merges both into `b12901045.json`

Both inference scripts read their settings (crop lengths, feature files, network width) from the checkpoint, so they need no other flags.

## Models

| | Task A | Task B |
|---|---|---|
| Model | Short-Chunk CNN | Dense head on frozen Whisper + MERT features |
| Input | log-mel of the mix | Whisper large-v3 (vocal stem, all layers) + Whisper language-ID probs + MERT-v1-330M (mix, all layers) |
| Validation top1 / top3 / score | 0.485 / 0.848 / 0.909 | 0.598 / 0.863 / 1.029 |

**Task A: Short-Chunk CNN** (`model.py`). Audio is resampled to 24 kHz and converted to a 128-band log-mel spectrogram. Seven conv blocks (Conv → BatchNorm → ReLU → MaxPool, 64 to 256 channels) and a global max pool give a 256-d embedding. A dense head (Linear → BatchNorm → ReLU → Dropout 0.5 → Linear) maps it to logits. Training uses three random 4 s crops per clip. At test time the logits of 10 evenly spaced 4 s crops are averaged.

**Task B: dense head on frozen features** (`train_features.py`). Clips are first split into vocals / accompaniment with Demucs (`htdemucs`). Each frozen model's hidden states are mean + std pooled over time, for every layer:

- Whisper large-v3 encoder on the vocal stem (33 layers, pooled over vocal-active frames), plus its language-ID probabilities
- MERT-v1-330M on the mix (25 layers)

A learned softmax weighting mixes each model's layers into one vector (SUPERB style). The vectors are concatenated and passed through the same dense head as the CNN.

## Files

| File | Purpose |
|---|---|
| `DeepMIR_hw1.ipynb` | End-to-end reproduction notebook (start here) |
| `inference.py` | Task A: test predictions from a `train.py` checkpoint |
| `inference_features.py` | Task B: test predictions from a `train_features.py` checkpoint |
| `model.py` | Short-Chunk CNN and the shared dense head |
| `dataset.py` | Audio → log-mel datasets and per-task config (`TASKS`) |
| `train.py` | Train the CNN |
| `separate.py` | Demucs vocal / accompaniment separation |
| `extract_features.py` | Extract all-layer Whisper / MERT features |
| `train_features.py` | Train the dense head on extracted features |
| `probe_layers.py` | Per-layer linear probe, used to analyse which layers carry the label |
| `b12901045.json` | Submitted predictions |

## Training from scratch

Run these from the `hw1/` folder after downloading the data. Training logs to [Weights & Biases](https://wandb.ai) (`wandb login` first).

**Task A**

```bash
python train.py --task year --random_crop --run_name crop_only
# best checkpoint: checkpoints/year/best_combined.pt
```

**Task B**

```bash
# 1. Source separation into separated_B/<sample_id>/{vocals,no_vocals}.wav
#    (demucs pins an old torchaudio, so install it without dependencies)
pip install --no-deps git+https://github.com/facebookresearch/demucs#egg=demucs
pip install julius lameenc openunmix dora-search
python separate.py --task market

# 2. Feature extraction into features/market/
python extract_features.py --extractor whisper --source vocals --tasks market
python extract_features.py --extractor mert    --source mix    --tasks market

# 3. Train the dense head on the manifest's train/validation split
python train_features.py --task market --lang_probs --folds 0 \
    --features features/market/large-v3_vocals.npz features/market/MERT-v1-330M_mix.npz
# checkpoint: checkpoints/market_features/<run name>.pt
```
