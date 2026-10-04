from __future__ import annotations

import random
import torch

from multilingual_transformer.configs.config import AppConfig
from multilingual_transformer.data.dataset import DataPipeline, TranslationDataset
from multilingual_transformer.engine.decoder import TranslationGenerator
from multilingual_transformer.engine.evaluator import TranslationEvaluator
from multilingual_transformer.engine.loader import build_model, fit_tokenizers, load_tokenizers
from multilingual_transformer.engine.trainer import Trainer
from multilingual_transformer.utils.distributed import barrier, run_auto_distributed
from multilingual_transformer.utils.helpers import free_memory, set_seed

import argparse
from pathlib import Path


def print_stats_table(
    config: AppConfig, n_params: int, train_loader_len: int, total_time: float,
    peak_mem: float, peak_res: float, bleu_greedy: float, chrf_greedy: float,
    bleu_beam: float, chrf_beam: float, src_vocab_sz: int, tgt_vocab_sz: int,
) -> None:
    print("\n" + "=" * 70)
    print("              CONSOLIDATED FINAL STATISTICS")
    print("=" * 70)
    print(f"Target Language   : {config.language.target_lang} ({config.language.tgt_lang})")
    print(f"Dataset Source    : {config.language.dataset_name}")
    print(f"Tokenizers Used   : SRC = {config.tokenizer.algo_src.upper()} | TGT = {config.tokenizer.algo_tgt.upper()}")
    print(f"Vocab Sizes       : SRC = {src_vocab_sz:,} | TGT = {tgt_vocab_sz:,}")
    print(f"Model Parameters  : {n_params:,} ({n_params * 4 / 1024**2:.1f} MB fp32)")
    print(f"Transformer Arch  : d_model {config.model.d_model} / num_layers {config.model.num_layers}/ d_ff {config.model.d_ff}")
    print("-" * 70)
    print(f"Epochs            : {config.training.epochs}")
    print(f"Train/Val/Test    : {config.data.train_size:,} / {config.data.val_size:,} / {config.data.test_size:,}")
    print(f"BlEU Samples      : {config.training.bleu_sample}")
    print(f"Batch Size        : {config.training.batch_size} (Train) | {config.training.gen_batch_size} (Eval)")
    print(f"Steps per Epoch   : {train_loader_len:,}")
    print(f"Total Train Time  : {total_time / 60:.2f} minutes")
    print(f"Peak GPU          : {peak_mem:.2f} GB")
    print(f"Peak Reserved GPU : {peak_res:.2f} GB")
    print("-" * 70)
    print(f"Greedy Decoding   -> BLEU: {bleu_greedy:5.2f} | CHRF: {chrf_greedy:5.2f}")
    print(f"Beam Decoding     -> BLEU: {bleu_beam:5.2f} | CHRF: {chrf_beam:5.2f}")
    print("=" * 70 + "\n")


def train_worker(rank: int, world_size: int, config_path: str) -> None:
    is_main = rank == 0
    cfg = AppConfig.from_toml(config_path)

    if is_main:
        print("\n" + "=" * 50)
        print("          INITIALIZING TRAINING PIPELINE")
        print("=" * 50)
        print(f"Dataset Name : {cfg.language.dataset_name}")
        print(f"Language     : {cfg.language.target_lang} ({cfg.language.tgt_lang})")
        print(f"Epochs       : {cfg.training.epochs}")
        print(f"Train Size   : {cfg.data.train_size}")
        print(f"Val Size     : {cfg.data.val_size}")
        print(f"Test Size    : {cfg.data.test_size}")
        print(f"Save Best    : {cfg.training.save_best}")
        print(f"GPUs Active  : {world_size}")
        print("=" * 50 + "\n")

    set_seed(cfg.project.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    pipeline = DataPipeline(
        dataset_name=cfg.language.dataset_name,
        src_lang=cfg.language.src_code,
        tgt_lang=cfg.language.tgt_lang,
        lang_pair=cfg.language.lang_pair,
        data_dir=cfg.data.data_dir,
        max_len=cfg.data.max_len,
        max_ratio=cfg.data.max_length_ratio,
        seed=cfg.project.seed,
    )
    sizes = (cfg.data.train_size, cfg.data.val_size, cfg.data.test_size)

    if is_main:
        tr_src, tr_tgt, val_src, val_tgt, te_src, te_tgt = pipeline.acquire_corpus(*sizes, cfg.data.force_download,verbose=True)
        print("Training tokenizers...")
        tok_src, tok_tgt = fit_tokenizers(cfg, tr_src, tr_tgt)
    barrier()

    if not is_main:
        tr_src, tr_tgt, val_src, val_tgt, te_src, te_tgt = pipeline.acquire_corpus(*sizes, False,verbose=False)
        tok_src, tok_tgt = load_tokenizers(cfg)

    tr_ds = TranslationDataset(tr_src, tr_tgt, tok_src, tok_tgt, cfg.data.max_len)
    val_ds = TranslationDataset(val_src, val_tgt, tok_src, tok_tgt, cfg.data.max_len)
    del tr_src, tr_tgt, val_src, val_tgt

    tr_loader = pipeline.create_loader(tr_ds, cfg.training.batch_size, shuffle=True, rank=rank, world_size=world_size, seed=cfg.project.seed)
    val_loader = pipeline.create_loader(val_ds, cfg.training.batch_size, shuffle=False, rank=rank, world_size=world_size, seed=cfg.project.seed)
    steps_per_epoch = len(tr_loader)
    free_memory()

    model = build_model(cfg, tok_src, tok_tgt)
    n_params = sum(p.numel() for p in model.parameters())

    trainer = Trainer(model, cfg, tr_loader, val_loader, tok_src, tok_tgt, rank=rank, world_size=world_size)
    train_time, peak_mem, peak_res = trainer.fit()
    
    barrier()   # rank 0 has finished writing checkpoints before anyone restores

    if is_main:
        print("\n" + "=" * 70)
        print("                    SAVED CHECKPOINTS")
        print("=" * 70)
        for label, info in (("Best (lowest val)", trainer.best_info), ("Final (last epoch)", trainer.last_info)):
            if info:
                print(f"{label:<19}: epoch {info['epoch']}/{cfg.training.epochs} | "
                      f"Train {info['train_loss']:.4f} | Val {info['val_loss']:.4f} | {info['path']}")
        print("=" * 70)

    trainer.restore()
    trainer.release()
    del trainer, tr_loader, val_loader, tr_ds, val_ds
    free_memory()

    # Multi-GPU Beam and Greedy Inference
    generator = TranslationGenerator(
        model, tok_src, tok_tgt, cfg.data.max_len, torch.device("cuda", torch.cuda.current_device()),
        no_repeat_ngram_size=cfg.inference.no_repeat_ngram_size,
    )
    evaluator = TranslationEvaluator(generator)
    gen_bs = cfg.training.gen_batch_size

    bleu_greedy, chrf_greedy, _ = evaluator.evaluate(
        te_src, te_tgt, method="greedy", batch_size=gen_bs, sample_size=cfg.training.bleu_sample,
        rank=rank, world_size=world_size,
    )
    bleu_beam, chrf_beam, _ = evaluator.evaluate(
        te_src, te_tgt, method="beam", beam_size=5, batch_size=gen_bs, sample_size=cfg.training.bleu_sample,
        rank=rank, world_size=world_size,
    )

    if is_main:
        print_stats_table(
            cfg, n_params, steps_per_epoch, train_time, peak_mem, peak_res,
            bleu_greedy, chrf_greedy, bleu_beam, chrf_beam,
            tok_src.get_vocab_size(), tok_tgt.get_vocab_size(),
        )

        print(f"\nBLEU signature: {evaluator.signatures.get('bleu')}")
        print(f"chrF signature: {evaluator.signatures.get('chrf')}")

        print("="*55)

        if cfg.inference.sample_source == "test":
            k = min(cfg.inference.num_samples, len(te_src))
            idx = random.Random(cfg.project.seed).sample(range(len(te_src)), k)
            sentences = [te_src[i] for i in idx]
            refs = [te_tgt[i] for i in idx]
            title = "Test"
        else:
            sentences = cfg.inference.sample_sentences
            refs = None
            title = "Config"

        if sentences:
            print(f"\n=== {title} Sample Translations ({cfg.language.lang_pair}) ===")
            greedy_preds = generator.batched_greedy_decode(sentences)
            beam_preds = generator.batched_beam_decode(sentences, beam_size=5)
            for i, (sent, g_pred, b_pred) in enumerate(zip(sentences, greedy_preds, beam_preds), 1):
                print(f"[{i}] EN   : {sent}")
                if refs is not None:
                    print(f"    Ref    : {refs[i - 1]}")
                print(f"    Greedy : {g_pred}")
                print(f"    Beam   : {b_pred}\n")


def _launch_train(rank: int, world_size: int) -> None:
    """Top-level wrapper so multiprocessing can pickle the function."""
    script_dir = Path(__file__).resolve().parent
    default_config = script_dir.parent / "configs" / "transformer_config.toml"
    
    parser = argparse.ArgumentParser(description="Transformer Engine")
    parser.add_argument("--config", type=str, default=str(default_config))
    args = parser.parse_args()
    
    train_worker(rank, world_size, args.config)

def main() -> None:
    run_auto_distributed(_launch_train)

if __name__ == "__main__":
    main()