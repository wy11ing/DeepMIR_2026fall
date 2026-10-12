from miditok import REMI, TokenizerConfig
from miditok.data_augmentation import augment_dataset
from pathlib import Path

TOKENIZER_PARAMS = {
    "pitch_range": (21, 109),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": True,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": False,
    "use_programs": True,
    "num_tempos": 32,  # number of tempo bins
    "tempo_range": (40, 250),  # (min, max)
}

def main():
    config = TokenizerConfig(**TOKENIZER_PARAMS)

    # create the tokenizer
    tokenizer = REMI(config)

    data_path = Path("POP909", "training").resolve()
    midi_aug_path = Path("POP909", "aug").resolve()

    augment_dataset(
        data_path, 
        pitch_offsets=[-12, 12],
        velocity_offsets=[-4, 5],
        duration_offsets=[-0.5, 1],
        out_path=midi_aug_path,
    )
    tokenizer.tokenize_dataset(        # 2 velocity and 1 duration values
        midi_aug_path,
        Path("POP909", "tok_merged").resolve()
    )


if __name__ == "__main__":
    main()
