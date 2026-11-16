#!/usr/bin/env bash
# Normalized-tag re-scoring + tag-style prompt experiment. Run ONE step per invocation; each step is a single
# process capped at 14 min (login-node limit is 15 min):
#   bash run_tagstyle.sh prompts      # rebuild test_prompts_tagstyle.csv, print samples           (~seconds)
#   bash run_tagstyle.sh rescore retr | cfg | apg7 | apg16   # per-tag MQA with normalized tags  (~3-5 min each)
#   bash run_tagstyle.sh gen          # APG s16 with tag-style prompts                             (~7 min)
#   bash run_tagstyle.sh evalnew      # caption + per-tag MQA of the new run (original captions)    (~8 min)
# Re-running a killed rescore/evalnew step resumes (fixed workdir + --mqa_resume).
set -euo pipefail
cd "$(dirname "$0")"
HW=$(cd .. && pwd)
MQA="--mqa_repo /work/b12901045/MQAScore --mqa_python /work/b12901045/MQAScore/.venv/bin/python \
     --mqa_model_id /work/b12901045/pretrained_models/music-flamingo-hf --mqa_resume"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
NEW=$HW/generated/apg_s16_tagstyle

case "${1:-}" in
  prompts)
    .venv/bin/python normalize_tags.py --inp test_tags.jsonl --out test_tags_norm.jsonl
    .venv/bin/python rewrite_prompts.py --tags test_tags_norm.jsonl --out test_prompts_tagstyle.csv
    sed -n '2,13p;23p;31p;43,44p;59p;65,67p' test_prompts_tagstyle.csv ;;
  rescore)
    declare -A DIRS=([retr]=$HW/retrieved_by_id [cfg]=$HW/generated/generated_v0
                     [apg7]=$HW/generated/apg_s7_m-0.5_e0.4_n12 [apg16]=$HW/generated/apg_s16_m-0.5_e0.4_n12)
    run=${2:?usage: rescore retr|cfg|apg7|apg16}
    timeout 840 .venv/bin/python evaluate.py --gen_dir "${DIRS[$run]}" --metrics mqa_tag --mqa_tags test_tags_norm.jsonl \
      $MQA --workdir "$PWD/eval_work/norm_$run" --out "eval_mqa_tagnorm_$run.json" ;;
  gen)
    timeout 840 .venv-cu124/bin/python batch_generate.py --csv test_prompts_tagstyle.csv --out_dir "$HW/generated" \
      --name apg_s16_tagstyle --apg --apg_space x0 --scales 16 --momentums -0.5 --etas 0.4 --norm_thresholds 12 --deadline 780 ;;
  evalnew)
    # --csv test.csv: score against the ORIGINAL captions, not the rewritten prompts
    timeout 840 .venv/bin/python evaluate.py --csv test.csv --gen_dir "$NEW" --metrics mqa,mqa_tag \
      --mqa_tags test_tags_norm.jsonl $MQA --workdir "$PWD/eval_work/apg16_tagstyle" --out eval_mqa_apg16_tagstyle.json ;;
  hybrid-prompts)
    .venv/bin/python rewrite_prompts.py --mode hybrid --tags test_tags_norm.jsonl --out test_prompts_hybrid.csv ;;
  hybrid-gen)
    timeout 840 .venv-cu124/bin/python batch_generate.py --csv test_prompts_hybrid.csv --out_dir "$HW/generated" \
      --name apg_s16_hybrid --apg --apg_space x0 --scales 16 --momentums -0.5 --etas 0.4 --norm_thresholds 12 --deadline 780 ;;
  hybrid-eval)
    timeout 840 .venv/bin/python evaluate.py --csv test.csv --gen_dir "$HW/generated/apg_s16_hybrid" --metrics mqa,mqa_tag \
      --mqa_tags test_tags_norm.jsonl $MQA --workdir "$PWD/eval_work/apg16_hybrid" --out eval_mqa_apg16_hybrid.json ;;
  boost-gen)   # cross-attention boost of instrument words, weight ${2:-2}
    timeout 840 .venv-cu124/bin/python boost_generate.py --out_dir "$HW/generated/apg_s16_boost_inst_w${2:-2}" \
      --weight "${2:-2}" --attrs instrument --deadline 780 ;;
  boost-eval)
    timeout 840 .venv/bin/python evaluate.py --csv test.csv --gen_dir "$HW/generated/apg_s16_boost_inst_w${2:-2}" --metrics mqa,mqa_tag \
      --mqa_tags test_tags_norm.jsonl $MQA --workdir "$PWD/eval_work/boost_inst_w${2:-2}" --out "eval_mqa_apg16_boost_inst_w${2:-2}.json" ;;
  hilo-gen)    # tags already judged present (P>=0.9) x DOWN, other instrument tags x UP:  hilo-gen [DOWN] [UP]
    D=${2:-0.5}; U=${3:-2}; TAG=$([ "$D$U" = 0.52 ] && echo hilo || echo "hilo_d${D}_u${U}")
    .venv/bin/python make_weights.py --scores eval_mqa_tagnorm_apg16.json --down "$D" --up "$U" --out "boost_weights_$TAG.jsonl"
    timeout 840 .venv-cu124/bin/python boost_generate.py --out_dir "$HW/generated/apg_s16_boost_$TAG" \
      --weights "boost_weights_$TAG.jsonl" --deadline 780 ;;
  hilo-eval)   # hilo-eval [DOWN] [UP]
    D=${2:-0.5}; U=${3:-2}; TAG=$([ "$D$U" = 0.52 ] && echo hilo || echo "hilo_d${D}_u${U}")
    timeout 840 .venv/bin/python evaluate.py --csv test.csv --gen_dir "$HW/generated/apg_s16_boost_$TAG" --metrics mqa,mqa_tag \
      --mqa_tags test_tags_norm.jsonl $MQA --workdir "$PWD/eval_work/boost_$TAG" --out "eval_mqa_apg16_boost_$TAG.json" ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac
