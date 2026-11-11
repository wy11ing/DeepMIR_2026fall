# HW3 sample code: classifier-free guidance for Stable Audio Open

## Setup
```bash
conda create -n deepmir_hw3 python=3.11 && conda activate deepmir_hw3
pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
`stabilityai/stable-audio-open-1.0` is gated on Hugging Face: accept its license, then run `huggingface-cli login`.
Generation needs about 9 GB of GPU memory (fp16).

## Files
| File | Description |
| --- | --- |
| `cfg.py` | **Exercise**: implement `cfg(noise_pred_uncond, noise_pred_cond, guidance_scale)`. |
| `generate.py` | The `StableAudioPipeline` sampling loop with the CFG step replaced by your `cfg`. |

## Exercise
1. Implement `cfg` in `cfg.py`.
2. Generate audio and compare with diffusers' own pipeline:
```bash
python generate.py --prompt "jazz, electric guitar, piano, relaxing" --out out.wav --check
```
With a correct implementation the difference is `0` (written in the same form as diffusers) or very small.

Note: Stable Audio Open is a v-prediction model, so the model outputs are v rather than noise. CFG is linear, so the
formula is the same.
