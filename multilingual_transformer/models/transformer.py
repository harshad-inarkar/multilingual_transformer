from __future__ import annotations

import torch
from torch import Tensor, nn

from multilingual_transformer.models.layers import DecoderLayer, EncoderLayer, TokenEmbedding


def make_src_mask(src: Tensor, pad_idx: int) -> Tensor:
    """(B, 1, 1, S) bool; True = real token."""
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: Tensor, pad_idx: int) -> Tensor:
    """(B, 1, T, T) bool; padding mask & causal mask."""
    seq_len = tgt.size(1)
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=tgt.device).tril()
    return pad_mask & causal


class TransformerEncoder(nn.Module):
    def __init__(
        self, vocab_size: int, max_len: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float, activation: str = "swiglu"
    ) -> None:
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout, activation=activation) for _ in range(num_layers)]
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self, vocab_size: int, max_len: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, dropout: float, activation: str = "swiglu"
    ) -> None:
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout, activation=activation) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, vocab_size, bias=False)

        # # Tie target input embedding and output projection weights
        # self.fc_out.weight = self.embed.emb.weight

    def forward(
        self,
        x: Tensor,
        enc_out: Tensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
        last_only: bool = False,
    ) -> Tensor:
        """`last_only` projects only the final position to the vocab (decoding speed-up)."""
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        if last_only:
            x = x[:, -1:, :]
        return self.fc_out(x)


class MultilingualTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        max_len: int = 96,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "swiglu",  # <-- Added
    ) -> None:
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, max_len, d_model, num_layers, num_heads, d_ff, dropout, activation=activation)
        self.decoder = TransformerDecoder(tgt_vocab_size, max_len, d_model, num_layers, num_heads, d_ff, dropout, activation=activation)
    
    def forward(
        self, src: Tensor, tgt: Tensor, src_mask: Tensor | None = None, tgt_mask: Tensor | None = None
    ) -> Tensor:
        enc_out = self.encoder(src, src_mask)
        return self.decoder(tgt, enc_out, src_mask, tgt_mask)

        