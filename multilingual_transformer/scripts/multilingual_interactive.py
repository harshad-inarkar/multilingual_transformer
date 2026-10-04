from __future__ import annotations

from pathlib import Path
import tomllib
import warnings

import torch

from multilingual_transformer.configs.config import AppConfig
from multilingual_transformer.data.tokenizer import TokenizerManager
from multilingual_transformer.engine.decoder import TranslationGenerator
from multilingual_transformer.models.transformer import MultilingualTransformer

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir.parent / "configs" / "multilingual_config.toml"
    
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
        print(f"\n[Error] Checkpoint not found at: {ckpt_path}\n")
        return

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

    generator = TranslationGenerator(
        model, shared_tok, shared_tok, cfg.data.max_len, device,
        no_repeat_ngram_size=cfg.inference.no_repeat_ngram_size,
    )

    print("\n" + "=" * 60)
    print("Universal Translation Matrix (Pivot Enabled)")
    print(f"Supported Target Codes: {', '.join(valid_langs)}")
    print("Type 'q' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            raw_input = input("Enter: ").strip()
            
            if not raw_input:
                continue
            if raw_input.lower() in ["q", "exit"]: 
                break

            if ":" not in raw_input:
                print("Invalid format. Please use 'lang: text' (e.g., 'mr: how are you?')")
                continue

            tgt_lang, text = [part.strip() for part in raw_input.split(":", 1)]
            tgt_lang = tgt_lang.lower()

            if tgt_lang not in valid_langs:
                print(f"Unsupported language code '{tgt_lang}'. Choose from: {valid_langs}")
                continue
            if not text:
                continue

            # Automated Pivot Logic for Indic <-> Indic
            if "en" in valid_langs and tgt_lang != "en" and not text.isascii():
                # Step 1: Indic -> English
                en_prefix = token_fmt.format("en")
                pivot_input = f"{en_prefix} {text}"
                pivot_text = generator.batched_beam_decode([pivot_input], beam_size=5)[0]
                print(f"[EN PIVOT]   : {pivot_text}")
                
                # Step 2: English -> Target Indic
                tgt_prefix = token_fmt.format(tgt_lang)
                final_input = f"{tgt_prefix} {pivot_text}"
                beam_out = generator.batched_beam_decode([final_input], beam_size=5)[0]
            else:
                # Direct Translation
                tgt_prefix = token_fmt.format(tgt_lang)
                final_input = f"{tgt_prefix} {text}"
                beam_out = generator.batched_beam_decode([final_input], beam_size=5)[0]
                
            print(f"[{tgt_lang.upper()} Beam]: {beam_out}\n")
            
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

if __name__ == "__main__":
    main()