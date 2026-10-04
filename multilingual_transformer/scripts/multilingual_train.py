from __future__ import annotations
from pathlib import Path
import tomllib
import warnings

import torch

from multilingual_transformer.configs.config import AppConfig
from multilingual_transformer.data.dataset import TranslationDataset
from multilingual_transformer.data.multilingual_dataset import MultilingualDataPipeline
from multilingual_transformer.data.multilingual_tokenizer import MultilingualTokenizerManager
from multilingual_transformer.engine.decoder import TranslationGenerator
from multilingual_transformer.engine.evaluator import TranslationEvaluator
from multilingual_transformer.engine.trainer import Trainer
from multilingual_transformer.models.transformer import MultilingualTransformer
from multilingual_transformer.utils.distributed import barrier, run_auto_distributed
from multilingual_transformer.utils.helpers import free_memory, set_seed
from multilingual_transformer.utils.reporting import print_signatures, print_multilingual_samples


def print_multilingual_stats_table(
    cfg: AppConfig, n_params: int, train_loader_len: int, total_time: float,
    peak_mem: float, peak_res: float, vocab_sz: int, results: dict[str, tuple[float, float]],
) -> None:
    print("\n" + "=" * 70)
    print("              CONSOLIDATED MULTILINGUAL STATISTICS")
    print("=" * 70)
    print("Dataset Strategy  : Bidirectional Augmentation with Target Prefixes")
    print(f"Tokenizers Used   : Shared Vocabulary ({cfg.tokenizer.algo_tgt.upper()})")
    print(f"Shared Vocab Size : {vocab_sz:,}")
    print(f"Model Parameters  : {n_params:,} ({n_params * 4 / 1024**2:.1f} MB fp32)")
    print(f"Transformer Arch  : d_model {cfg.model.d_model} / num_layers {cfg.model.num_layers}/ d_ff {cfg.model.d_ff}")
    print("-" * 70)
    print(f"Epochs            : {cfg.training.epochs}")
    print(f"BlEU Samples      : {cfg.training.bleu_sample} (per pair)")
    print(f"Steps per Epoch   : {train_loader_len:,}")
    print(f"Total Train Time  : {total_time / 60:.2f} minutes")
    print(f"Peak GPU Memory   : {peak_res:.2f} GB Reserved / {peak_mem:.2f} GB Allocated")
    print("-" * 70)
    print("Final Target Benchmarks (Beam Decoding):")
    for pair, (bleu, chrf) in results.items():
        print(f"  {pair.upper():<7} -> BLEU: {bleu:5.2f} | CHRF: {chrf:5.2f}")
    print("=" * 70 + "\n")


def multilingual_worker(rank: int, world_size: int, config_path: str) -> None:
    is_main = rank == 0

    with open(config_path, "rb") as f:
        raw_cfg = tomllib.load(f)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = AppConfig.from_toml(config_path)

    set_seed(cfg.project.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    train_pairs = raw_cfg["multilingual"]["train_languages_pairs"]
    token_fmt = raw_cfg["multilingual"]["target_tokens_format"]

    unique_langs = set()
    for pair in train_pairs:
        src_l, tgt_l = pair.split("-")
        unique_langs.update([src_l, tgt_l])
    target_tokens = [token_fmt.format(lang) for lang in sorted(unique_langs)]

    target_langs = {pair.split("-")[1] for pair in train_pairs}
    active_dataset_map = {lang: raw_cfg["dataset_map"][lang] for lang in target_langs}

    if is_main:
        print("\n" + "=" * 50)
        print("      INITIALIZING MULTILINGUAL PIPELINE")
        print("=" * 50)
        print(f"Base Pairs   : {train_pairs}")
        print(f"Gen. Tokens  : {target_tokens}")
        print(f"Tokenizers   : Shared {cfg.tokenizer.algo_tgt.upper()} (Vocab: {cfg.tokenizer.max_vocab_size})")
        print(f"Epochs       : {cfg.training.epochs}")
        print(f"Save Best    : {cfg.training.save_best}")
        print(f"GPUs Active  : {world_size}")
        print("=" * 50 + "\n")

    pipeline = MultilingualDataPipeline(
        dataset_map=active_dataset_map,
        data_dir=cfg.data.data_dir,
        max_len=cfg.data.max_len,
        max_ratio=cfg.data.max_length_ratio,
        seed=cfg.project.seed,
    )

    shared_tok_path = cfg.data.tokenizer_dir / f"multilingual_shared_{cfg.tokenizer.max_vocab_size}.json"

    if is_main:
        cfg.data.data_dir.mkdir(parents=True, exist_ok=True)
        corpus = pipeline.acquire_multilingual_corpus(
            pairs_per_lang=raw_cfg["multilingual"]["pairs_per_lang"],
            val_per_lang=raw_cfg["multilingual"]["val_size_per_lang"],
            test_per_lang=raw_cfg["multilingual"]["test_size_per_lang"],
            force_download=cfg.data.force_download,
            verbose=True,
        )
            
        print("Training shared multilingual tokenizer...")
        combined_texts = corpus["train_src"] + corpus["train_tgt"]
        shared_tok = MultilingualTokenizerManager.train(
            combined_texts, cfg.tokenizer.algo_tgt, cfg.tokenizer.max_vocab_size, lang_tokens=target_tokens
        )
        cfg.data.tokenizer_dir.mkdir(parents=True, exist_ok=True)
        MultilingualTokenizerManager.save(shared_tok, shared_tok_path)
        
    barrier()

    if not is_main:
        corpus = pipeline.acquire_multilingual_corpus(
            pairs_per_lang=raw_cfg["multilingual"]["pairs_per_lang"],
            val_per_lang=raw_cfg["multilingual"]["val_size_per_lang"],
            test_per_lang=raw_cfg["multilingual"]["test_size_per_lang"],
            force_download=False,
            verbose=False,
        )
        shared_tok = MultilingualTokenizerManager.load(shared_tok_path)

    tr_ds = TranslationDataset(corpus["train_src"], corpus["train_tgt"], shared_tok, shared_tok, cfg.data.max_len)
    va_ds = TranslationDataset(corpus["val_src"], corpus["val_tgt"], shared_tok, shared_tok, cfg.data.max_len)

    tr_loader = pipeline.create_loader(tr_ds, cfg.training.batch_size, shuffle=True, rank=rank, world_size=world_size, seed=cfg.project.seed)
    val_loader = pipeline.create_loader(va_ds, cfg.training.batch_size, shuffle=False, rank=rank, world_size=world_size, seed=cfg.project.seed)
    free_memory()

    vocab_sz = shared_tok.get_vocab_size()
    model = MultilingualTransformer(
        src_vocab_size=vocab_sz, tgt_vocab_size=vocab_sz, max_len=cfg.data.max_len,
        d_model=cfg.model.d_model, num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads, d_ff=cfg.model.d_ff, dropout=cfg.model.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())

    trainer = Trainer(model, cfg, tr_loader, val_loader, shared_tok, shared_tok, rank=rank, world_size=world_size)
    train_time, peak_mem, peak_res = trainer.fit()
    barrier()
    trainer.restore()
    trainer.release()
    free_memory()

    generator = TranslationGenerator(
        model, shared_tok, shared_tok, cfg.data.max_len, torch.device("cuda", torch.cuda.current_device()),
        no_repeat_ngram_size=cfg.inference.no_repeat_ngram_size,
    )
    evaluator = TranslationEvaluator(generator)

    results = {}
    eval_directions = []
    for pair in train_pairs:
        l1, l2 = pair.split("-")
        eval_directions.extend([pair, f"{l2}-{l1}"])

    for pair_key in eval_directions:
        split = corpus["eval_splits"].get(pair_key)
        if split:
            tgt_prefix = split.get("tgt_prefix")
            bleu, chrf, _ = evaluator.evaluate(
                split["test_src"], split["test_tgt"], method="beam", beam_size=5,
                batch_size=cfg.training.gen_batch_size, sample_size=cfg.training.bleu_sample,
                rank=rank, world_size=world_size, tgt_prefix_token=tgt_prefix
            )
            results[pair_key] = (bleu, chrf)

    if is_main:
        print_multilingual_stats_table(cfg, n_params, len(tr_loader), train_time, peak_mem, peak_res, vocab_sz, results)
        print_signatures(evaluator)
        print_multilingual_samples(cfg, generator, corpus["eval_splits"], sorted(target_langs), sorted(unique_langs), token_fmt)


def _launch_multilingual(rank: int, world_size: int) -> None:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir.parent / "configs" / "multilingual_config.toml"
    multilingual_worker(rank, world_size, str(config_path))

def main() -> None:
    run_auto_distributed(_launch_multilingual)

if __name__ == "__main__":
    main()