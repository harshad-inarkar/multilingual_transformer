from __future__ import annotations

from typing import Any

import torch

from multilingual_transformer.configs.config import AppConfig
from multilingual_transformer.data.tokenizer import TokenizerManager
from multilingual_transformer.models.transformer import MultilingualTransformer


def fit_tokenizers(cfg: AppConfig, tr_src: list[str], tr_tgt: list[str]) -> tuple[Any, Any]:
    """Train source/target tokenizers and cache them next to the checkpoint."""
    tok_src = TokenizerManager.train(tr_src, cfg.tokenizer.algo_src, cfg.tokenizer.max_vocab_size)
    tok_tgt = TokenizerManager.train(tr_tgt, cfg.tokenizer.algo_tgt, cfg.tokenizer.max_vocab_size)
    TokenizerManager.save(tok_src, cfg.tokenizer_path("src"))
    TokenizerManager.save(tok_tgt, cfg.tokenizer_path("tgt"))
    return tok_src, tok_tgt


def load_tokenizers(cfg: AppConfig) -> tuple[Any, Any]:
    """Load the exact tokenizers used for training (retraining could change the vocab)."""
    paths = (cfg.tokenizer_path("src"), cfg.tokenizer_path("tgt"))
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Tokenizer not found at {p}. Re-run training to generate it.")
    return TokenizerManager.load(paths[0]), TokenizerManager.load(paths[1])


def build_model(cfg: AppConfig, tok_src: Any, tok_tgt: Any) -> MultilingualTransformer:
    return MultilingualTransformer(
        src_vocab_size=tok_src.get_vocab_size(),
        tgt_vocab_size=tok_tgt.get_vocab_size(),
        max_len=cfg.data.max_len,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
    )


def load_model(cfg: AppConfig, tok_src: Any, tok_tgt: Any, device: torch.device, kind: str | None = None) -> MultilingualTransformer:
    model = build_model(cfg, tok_src, tok_tgt)
    ckpt = torch.load(cfg.checkpoint_path(kind), map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    del ckpt
    return model.to(device).eval()
