from __future__ import annotations

from pathlib import Path
import tomllib
import warnings

import ipywidgets as widgets
import torch

from multilingual_transformer.configs.config import AppConfig
from multilingual_transformer.data.tokenizer import TokenizerManager
from multilingual_transformer.engine.decoder import TranslationGenerator
from multilingual_transformer.models.transformer import MultilingualTransformer

class MultilingualTranslationWidget:
    def __init__(self, generator: TranslationGenerator, valid_langs: list[str], token_fmt: str) -> None:
        self.generator = generator
        self.valid_langs = valid_langs
        self.token_fmt = token_fmt

    def render(self) -> widgets.VBox:
        src_dropdown = widgets.Dropdown(options=self.valid_langs, value="en", description="Source:")
        tgt_dropdown = widgets.Dropdown(options=self.valid_langs, value="hi", description="Target:")
        
        text_box = widgets.Text(
            value="education is essential for every child",
            description="Input:",
            layout=widgets.Layout(width="500px"),
        )
        method_dropdown = widgets.Dropdown(options=["greedy", "beam"], value="beam", description="Method:")
        translate_btn = widgets.Button(description="Translate", button_style="primary")
        out = widgets.Output()

        def on_click(_: widgets.Button) -> None:
            out.clear_output()
            with out:
                if src_dropdown.value == tgt_dropdown.value:
                    print("Source and Target languages must be different.")
                    return
                
                tgt_lang = tgt_dropdown.value.lower()
                tgt_prefix = self.token_fmt.format(tgt_lang)
                text = text_box.value
                
                # 1. Add tag to the text (For the Encoder)
                prefixed_text = f"{tgt_prefix} {text}"
                
                if method_dropdown.value == "greedy":
                    res = self.generator.greedy_decode(prefixed_text, tgt_prefix_token=tgt_prefix)
                else:
                    res = self.generator.batched_beam_decode([prefixed_text], beam_size=5, tgt_prefix_token=tgt_prefix)[0]
                    
                print(f"[{src_dropdown.value.upper()}] : {text}")
                print(f"[{tgt_dropdown.value.upper()}] : {res}")

        translate_btn.on_click(on_click)
        
        lang_selectors = widgets.HBox([src_dropdown, tgt_dropdown])
        return widgets.VBox([lang_selectors, text_box, method_dropdown, translate_btn, out])

def launch(config_path: str | None = None) -> widgets.VBox | None:
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent.parent / "configs" / "multilingual_config.toml")

    with open(config_path, "rb") as f:
        raw_cfg = tomllib.load(f)
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = AppConfig.from_toml(config_path)

    train_pairs = raw_cfg["multilingual"]["train_languages_pairs"]
    token_fmt = raw_cfg["multilingual"]["target_tokens_format"]
    valid_langs = set()
    for pair in train_pairs:
        valid_langs.update(pair.split("-"))
    valid_langs = sorted(list(valid_langs))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok_path = cfg.data.tokenizer_dir / f"multilingual_shared_{cfg.tokenizer.max_vocab_size}.json"
    ckpt_path = cfg.checkpoint_path("best" if cfg.training.save_best else "last")

    if not ckpt_path.exists():
        print(f"[Error] Checkpoint not found at: {ckpt_path}")
        print("Please run training first: !python -m multilingual_transformer.scripts.multilingual_train")
        return None

    print("Loading shared tokenizer and model weights (this takes a few seconds)...")
    shared_tok = TokenizerManager.load(tok_path)
    vocab_sz = shared_tok.get_vocab_size()

    model = MultilingualTransformer(
        src_vocab_size=vocab_sz, tgt_vocab_size=vocab_sz, max_len=cfg.data.max_len,
        d_model=cfg.model.d_model, num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads, d_ff=cfg.model.d_ff, dropout=cfg.model.dropout,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    print("Ready! Rendering widget...\n")
    generator = TranslationGenerator(
        model, shared_tok, shared_tok, cfg.data.max_len, device,
        no_repeat_ngram_size=cfg.inference.no_repeat_ngram_size,
    )
    return MultilingualTranslationWidget(generator, valid_langs, token_fmt).render()