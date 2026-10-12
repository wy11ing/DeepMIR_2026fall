# Usage For Each Scripts
### `tokenization.py`
This script is for processing the POP909 data and converting the MIDI messages into self-defined tokens
```bash
python tokenization.py --dataset_path ${POP909 training set path} --tok2idx_output ${token dictionary output path} --sequences_output ${tokenized song sequences output path} --pos_per_beat ${quantization setting}

# e.g. python tokenization.py --dataset_path ./POP909/training --tok2idx_output ./tok2idx.json --sequences_output ./all_song_sequences.json --pos_per_beat 16
```

### `eval_basic.py`
This script is for running objective metrics for `Basic Task a.`
```bash
python eval_basic.py --tok2idx_path ${token dictionary path} --sequences_path ${tokenized song sequences path} --output_csv ${results output path}

# e.g. for calculating the real data scores: python eval_basic.py --tok2idx_path ./tok2idx.json --sequences_path ./all_song_sequences.json --output_csv ./unconditional_metrics.csv
```

### `eval_bonus.py`
This script is for running objective metrics for `Bonus Task`
```bash
python eval_bonus.py --prompt_dir ${melody prompts folder path} --gen_midi_dir ${generated MIDIs folder path} --output_csv ${results output path}

# e.g. python eval_bonus.py --prompt_dir ./POP909/prompts/bonus --gen_midi_dir ./bonus_task_gen_midis --output_csv ./conditional_metrics.csv
```