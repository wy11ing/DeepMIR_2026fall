"""
BART for the seq2seq accompaniment model, with two optional fixes against the decoder ignoring the encoder
(diag_s2s.py: the plain model's validation loss was the same with the real melody, another song's, or none).
Both are off by default, so the plain model is exactly HF's BartForConditionalGeneration.

  bar_emb (A)       every encoder and decoder token gets a learned embedding of the bar it is in, so piano
                    tokens of bar i can find the melody of bar i through cross-attention from the start
  hist_dropout (B)  during training, this fraction of the decoder's input tokens (not Bar tokens, not the
                    labels) is replaced by PAD, whose embedding is zero; the piano history becomes less
                    reliable, so predicting it well needs the melody

Both settings live in the model config, so generation and diag_s2s.py pick them up from the checkpoint.
"""
import torch
from torch import nn
from transformers import BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right

from accomp_data import BOS, MAX_BARS, PAD, VOCAB

BAR0 = VOCAB["Bar_0"]
NO_BAR = MAX_BARS  # bar index of tokens before the first Bar token (BOS)

def bar_index(ids):
    """(B, L) token ids -> (B, L) index of the bar each token is in (Bar_i counts as bar i)."""
    is_bar = (ids >= BAR0) & (ids < BAR0 + MAX_BARS)
    idx = torch.where(is_bar, ids - BAR0, torch.full_like(ids, -1)).cummax(dim=1).values
    return torch.where(idx < 0, torch.full_like(idx, NO_BAR), idx)

class AccompBart(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        config.bar_emb = getattr(config, "bar_emb", False)
        config.hist_dropout = getattr(config, "hist_dropout", 0.0)
        if config.bar_emb:
            self.bar_embed = nn.Embedding(MAX_BARS + 1, config.d_model)
            nn.init.normal_(self.bar_embed.weight, std=config.init_std)

    def embed(self, embed_tokens, ids, bar_ids=None):
        if bar_ids is None:
            bar_ids = bar_index(ids)
        return embed_tokens(ids) + self.bar_embed(bar_ids)

    def encode(self, input_ids, attention_mask=None):
        encoder = self.get_encoder()
        if not self.config.bar_emb:
            return encoder(input_ids=input_ids, attention_mask=attention_mask)
        return encoder(inputs_embeds=self.embed(encoder.embed_tokens, input_ids), attention_mask=attention_mask)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None,
                encoder_outputs=None, decoder_bar_ids=None, **kwargs):
        """
        decoder_bar_ids: bar index of each decoder input token. Only needed when decoding with a cache, where
        the new tokens alone don't show which bar they are in; otherwise it is read from decoder_input_ids.
        """
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = shift_tokens_right(labels, PAD, BOS)
        if decoder_bar_ids is None:
            decoder_bar_ids = bar_index(decoder_input_ids)  # from the clean ids, before dropout
        if self.training and self.config.hist_dropout > 0:
            keep_bar = (decoder_input_ids >= BAR0) & (decoder_input_ids < BAR0 + MAX_BARS)
            drop = (torch.rand(decoder_input_ids.shape, device=decoder_input_ids.device) < self.config.hist_dropout)
            drop &= ~keep_bar & (decoder_input_ids != BOS)
            decoder_input_ids = decoder_input_ids.masked_fill(drop, PAD)
        if encoder_outputs is None:
            encoder_outputs = self.encode(input_ids, attention_mask)
        if self.config.bar_emb:
            embeds = self.embed(self.get_decoder().embed_tokens, decoder_input_ids, decoder_bar_ids)
            return super().forward(attention_mask=attention_mask, decoder_inputs_embeds=embeds, labels=labels,
                                   encoder_outputs=encoder_outputs, **kwargs)
        return super().forward(attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, labels=labels,
                               encoder_outputs=encoder_outputs, **kwargs)
