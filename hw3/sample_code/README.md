# HW3: Guidance and conditioning for Stable Audio Open

The code generates music from text with Stable Audio Open 1.0. The sampling loop is written out by hand, so the
guidance step can be swapped:

- **CFG**: classifier-free guidance
- **APG**: adaptive projected guidance
- **Cross-attention reweighting** of individual prompt words

The output is scored against the captions with FAD, KL, CLAP and **MQAScore** (Music Flamingo yes/no judge).

## Setup
```bash
conda create -n deepmir_hw3 python=3.11 && conda activate deepmir_hw3
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
- `stabilityai/stable-audio-open-1.0` is gated on Hugging Face. Accept its license, then run `huggingface-cli login`.
- Generation needs about 9 GB of GPU memory (fp16).
- **On a machine whose driver only supports CUDA 12.4** (the cluster used here), the cu128 wheels cannot use the GPU.
  Generation used a separate env, `.venv-cu124`, with `torch==2.6.0+cu124` and the other requirements.
  Evaluation (`evaluate.py`) runs from `.venv`.
- **MQAScore** lives in its own repo and env (`/work/b12901045/MQAScore` with its `.venv`). It uses the judge
  checkpoint `/work/b12901045/pretrained_models/music-flamingo-hf`. `evaluate.py` calls it as a subprocess through
  `--mqa_repo`, `--mqa_python` and `--mqa_model_id`.

## Directory layout
```
hw3/
├── sample_code/            # this folder: all scripts, prompt/tag files

```

## Scripts

### Generation
| File | What it does |
| --- | --- |
| `cfg.py` | Guidance rules. `cfg()` is classifier-free guidance. `apg()` with `MomentumBuffer`/`project` is **the** APG implementation (Sadat et al., 2024): momentum, norm clipping and the parallel/orthogonal split. |
| `generate.py` | The `StableAudioPipeline` sampling loop with the guidance step taken out. `generate()` uses `cfg`, or `apg` when `use_apg=True`. `apg_space="v"` (default) guides the v-prediction; `"x0"` guides the denoised estimate. CLI generates one prompt. |
| `batch_generate.py` | Generates every caption in a CSV, optionally sweeping scale, momentum, eta and norm threshold (grid). Writes `<out_dir>/<tag>/<id>.wav` + `config.json`. Skips existing files, so a rerun resumes. Supports `--name`, `--ids` and `--deadline`. |
| `boost_generate.py` | APG s16 (x0) + **cross-attention word reweighting**. Chosen words' T5 tokens get their attention multiplied by w in every DiT block (log w is added to the attention logits). The words are either `--attrs`/`--weight` (e.g. all instrument tags ×2) or per-word weights from `--weights`. Logs the reweighted tokens to `boost_tokens.jsonl`. |
| `make_weights.py` | Builds a `--weights` file from a previous run's per-tag scores. Tags already judged present (P ≥ 0.9) get ×`--down`; the other instrument tags get ×`--up`. ⚠ This uses test-set judge scores, so treat it as an oracle experiment. |

### Prompts and tags
| File | What it does |
| --- | --- |
| `test.csv` | The 100 test captions (`id, caption`). |
| `extract_tags_hf.py` | Extracts MQAScore tags (genre, instrument, mood_theme, …) from the captions with Qwen3-4B-Instruct (HF transformers port of MQAScore's `extract_tags.py`). Run it with the MQAScore env. Output: `test_tags.jsonl`. |
| `normalize_tags.py` | Canonical tag spellings (`drummachine` → `drum machine`, `hiphop` → `hip hop`, …). Output: `test_tags_norm.jsonl`, which all experiments use. |
| `rewrite_prompts.py` | Rule-based prompt rewriting for generation only. `--mode tagstyle` makes a comma tag list. `--mode hybrid` keeps the caption and adds a prefix such as `High energy, driving rhythm, 124 BPM.`. Outputs: `test_prompts_tagstyle.csv`, `test_prompts_hybrid.csv`. |

### Retrieval and evaluation
| File | What it does |
| --- | --- |
| `download.py` | Downloads the shared data folder from Google Drive (`gdown`). |
| `clap_retrieve.py` | Text-to-music retrieval with LAION-CLAP. Finds the closest real clip in `--audio_dir` for each caption and writes `retrieval.json`. That retrieved audio is the FAD/KL reference and the "real music" baseline. |
| `evaluate.py` | Metrics `fad`, `kl`, `clap`, `mqa` (caption protocol) and `mqa_tag` (per-tag protocol, needs `--mqa_tags`). Writes `--out` JSON with a summary and per-item scores. `--mqa_resume` continues a killed MQA run (use a fixed absolute `--workdir`). |
| `run_tagstyle.sh` | Driver for all the experiments. Each step is **one process under `timeout 840`**, so nothing exceeds the 15-min login-node limit. |

## How to use

All commands run from `sample_code/`. On the cluster, use `.venv-cu124/bin/python` for generation and
`.venv/bin/python` for evaluation.

### 1. CFG exercise (sanity check against diffusers)
```bash
python generate.py --prompt "jazz, electric guitar, piano, relaxing" --out out.wav --check
```
With a correct `cfg` the difference is `0` (written in the same form as diffusers) or very small.
Stable Audio Open is a v-prediction model, so its outputs are v rather than noise. CFG is linear, so the formula
is the same.

### 2. Generate the test set
```bash
# CFG, scale 7
python batch_generate.py --csv test.csv --out_dir ../generated --scales 7
# APG (v space), sweep
python batch_generate.py --out_dir ../generated --apg --scales 7,10 --momentums 0,-0.5 --etas 0,0.4 --norm_thresholds 0,12
# APG s16 in x0 space, fixed folder name, stop after 13 min (rerun the same command to resume)
python batch_generate.py --csv test.csv --out_dir ../generated --name apg_s16_m-0.5_e0.4_n12 \
    --apg --apg_space x0 --scales 16 --momentums -0.5 --etas 0.4 --norm_thresholds 12 --deadline 780
```
100 clips take about 7 min on an H100 (≈4 s per clip plus 30 s to load the model).

### 3. Word reweighting
```bash
# boost every instrument word of each caption x2
python boost_generate.py --out_dir ../generated/apg_s16_boost_inst_w2 --weight 2 --attrs instrument --deadline 780
# per-word weights from a previous run's tag scores (oracle): strong tags x0.8, weak instrument tags x2
python make_weights.py --scores eval_mqa_tagnorm_apg16.json --down 0.8 --up 2 --out boost_weights_hilo_d0.8_u2.jsonl
python boost_generate.py --out_dir ../generated/apg_s16_boost_hilo_d0.8_u2 --weights boost_weights_hilo_d0.8_u2.jsonl
```
With `--weight 1`, the output is bit-identical to plain APG.

### 4. Evaluate
```bash
MQA="--mqa_repo /work/b12901045/MQAScore --mqa_python /work/b12901045/MQAScore/.venv/bin/python \
     --mqa_model_id /work/b12901045/pretrained_models/music-flamingo-hf --mqa_resume"
# FAD / KL / CLAP against the retrieved real music
python evaluate.py --gen_dir ../generated/<run> --ref_dir ../audio_mp3 --retrieval retrieval.json
# MQAScore, caption + per-tag (always score against the ORIGINAL captions, even for rewritten prompts)
python evaluate.py --csv test.csv --gen_dir ../generated/<run> --metrics mqa,mqa_tag --mqa_tags test_tags_norm.jsonl \
    $MQA --workdir "$PWD/eval_work/<run>" --out eval_mqa_<run>.json
```
Scoring 100 clips with both MQA protocols takes about 3–4 min. `--workdir` must be an absolute path.

### 5. Reproduce the experiments
`run_tagstyle.sh` runs one step per invocation:
```bash
bash run_tagstyle.sh prompts                    # normalize tags + build tag-style prompts
bash run_tagstyle.sh rescore apg16              # per-tag MQA of an existing run with normalized tags (retr|cfg|apg7|apg16)
bash run_tagstyle.sh gen      && bash run_tagstyle.sh evalnew       # tag-style prompts
bash run_tagstyle.sh hybrid-prompts
bash run_tagstyle.sh hybrid-gen && bash run_tagstyle.sh hybrid-eval # hybrid prompts
bash run_tagstyle.sh boost-gen 2 && bash run_tagstyle.sh boost-eval 2           # instrument words x2
bash run_tagstyle.sh hilo-gen 0.8 2 && bash run_tagstyle.sh hilo-eval 0.8 2     # strong x0.8 / weak x2
```

## Notes and pitfalls
- **Determinism:** the sampler (CosineDPMSolver SDE) is reproducible with the same seed on the same GPU type, but
  not across hardware. `apg_s7_m-0.5_e0.4_n12/` was generated elsewhere and can't be reproduced bit-for-bit here.
- **MQAScore caveats**:
  - The caption protocol is biased toward "yes": silence scores 0.79.
  - Padding 10 s clips to 30 s inflates the score by about 0.09.
  - Per-tag **genre** and **instrument** questions are the reliable signals; mood and context are yes-biased.
  - Compare runs paired, on the same prompts.
- MQAScore's `score.py --fix-seconds full` crashes; use a number of seconds.
- Reweighting and prompt rewriting change only the **generation** input. Evaluation always uses `test.csv` and
  `test_tags_norm.jsonl`.
