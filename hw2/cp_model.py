import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

# Embedding size per token type, following the Compound Word Transformer
# (tempo 128, chord 256, bar-beat 64, type 32, pitch 512, duration 128, velocity 128).
# Program is not in the original; POP909 only uses program 0, so it gets a small embedding.
EMB_SIZES = {
    "Family": 32,
    "Position": 64,  # also holds Bar (miditok puts Bar and Position in the same slot)
    "Pitch": 512,
    "Velocity": 128,
    "Duration": 128,
    "Program": 32,
    "Chord": 256,
    "Tempo": 128,
}

class CPTransformer(nn.Module):
    """
    Compound Word Transformer (Hsiao et al., 2021) on top of a GPT-2 backbone.

    Input: compound tokens of shape (B, L, K), one id per token type (K types, Family first).
    - Each type has its own embedding; they are concatenated and projected to n_embd.
    - The backbone is a causal transformer (GPT-2 softmax attention instead of the paper's linear attention).
    - Two-stage output: Family (the paper's "type") is predicted from the hidden state h first; then
      h is concatenated with the Family embedding and projected, and every other type is predicted from that.
      During training the *target* Family is used (teacher forcing); during sampling the sampled one.
    """
    def __init__(self, vocab_sizes, emb_sizes, n_positions=512, n_layer=12, n_head=8, n_embd=512, dropout=0.1):
        super().__init__()
        self.config = dict(vocab_sizes=vocab_sizes, emb_sizes=emb_sizes, n_positions=n_positions,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout)
        self.embs = nn.ModuleList(nn.Embedding(v, e) for v, e in zip(vocab_sizes, emb_sizes))
        self.in_proj = nn.Linear(sum(emb_sizes), n_embd)
        # vocab_size=1: GPT-2's own token embedding is unused (we pass inputs_embeds); it keeps the positional one
        self.backbone = GPT2Model(GPT2Config(
            vocab_size=1, n_positions=n_positions, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout, bos_token_id=None, eos_token_id=None,
        ))
        self.family_head = nn.Linear(n_embd, vocab_sizes[0])
        self.family_proj = nn.Linear(n_embd + emb_sizes[0], n_embd)
        self.heads = nn.ModuleList(nn.Linear(n_embd, v) for v in vocab_sizes[1:])

    def hidden(self, x):
        """(B, L, K) ids -> (B, L, n_embd) causal hidden states."""
        emb = torch.cat([emb(x[..., k]) for k, emb in enumerate(self.embs)], dim=-1)
        return self.backbone(inputs_embeds=self.in_proj(emb)).last_hidden_state

    def family_logits(self, h):
        return self.family_head(h)

    def other_logits(self, h, family_ids):
        """Logits for types 1..K-1, conditioned on the (target or sampled) Family ids."""
        y = self.family_proj(torch.cat([h, self.embs[0](family_ids)], dim=-1))
        return [head(y) for head in self.heads]

    def forward(self, input_ids, pad_id=0):
        """Next-compound-token loss. Returns (mean loss over types, list of per-type losses)."""
        x, target = input_ids[:, :-1], input_ids[:, 1:]
        h = self.hidden(x)
        logits = [self.family_logits(h)] + self.other_logits(h, target[..., 0])
        # Ignore_None slots are predicted too (as in the original), so the model learns which
        # types a Family uses; only padding is excluded
        losses = [F.cross_entropy(lg.flatten(0, 1), target[..., k].flatten(), ignore_index=pad_id)
                  for k, lg in enumerate(logits)]
        return torch.stack(losses).mean(), losses

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        json.dump(self.config, open(path / "config.json", "w"), indent=2)
        torch.save(self.state_dict(), path / "model.pt")

    @classmethod
    def from_pretrained(cls, path, map_location="cpu"):
        path = Path(path)
        model = cls(**json.load(open(path / "config.json")))
        model.load_state_dict(torch.load(path / "model.pt", map_location=map_location))
        return model
