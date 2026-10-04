from __future__ import annotations

import random
from typing import Any, Iterable


def print_checkpoint_table(cfg: Any, trainer: Any) -> None:
    """Best and final epoch with their losses. Only rank 0 saves checkpoints, so only call it on rank 0."""
    print("\n" + "=" * 70)
    print("                    SAVED CHECKPOINTS")
    print("=" * 70)
    for label, info in (("Best (lowest val)", trainer.best_info), ("Final (last epoch)", trainer.last_info)):
        if info:
            print(f"{label:<19}: epoch {info['epoch']}/{cfg.training.epochs} | "
                  f"Train {info['train_loss']:.4f} | Val {info['val_loss']:.4f} | {info['path']}")
    print(f"Used for evaluation: {'best' if cfg.training.save_best else 'final'}")
    print("=" * 70)


def print_signatures(evaluator: Any) -> None:
    """sacreBLEU signatures of the last evaluation, so scores are reproducible."""
    print(f"\nBLEU signature: {evaluator.signatures.get('bleu')}")
    print(f"chrF signature: {evaluator.signatures.get('chrf')}")


def print_translation_samples(cfg: Any, generator: Any, te_src: list[str], te_tgt: list[str], beam_size: int = 5) -> None:
    """One-to-one model: sample_source = "test" prints seeded random test pairs, "config" prints sample_sentences."""
    if cfg.inference.sample_source == "test":
        k = min(cfg.inference.num_samples, len(te_src))
        # Seeded local RNG: same samples every run, without touching global random state
        idx = random.Random(cfg.project.seed).sample(range(len(te_src)), k)
        sentences = [te_src[i] for i in idx]
        refs: list[str] | None = [te_tgt[i] for i in idx]
        title = "Test"
    else:
        sentences = list(cfg.inference.sample_sentences)
        refs = None
        title = "Config"

    print(f"\n=== {title} Sample Translations ({cfg.language.lang_pair}) ===")
    if not sentences:
        print("(no samples configured)\n")
        return
    greedy_preds = generator.batched_greedy_decode(sentences)
    beam_preds = generator.batched_beam_decode(sentences, beam_size=beam_size)
    for i, (sent, g_pred, b_pred) in enumerate(zip(sentences, greedy_preds, beam_preds), 1):
        print(f"[{i}] EN   : {sent}")
        if refs is not None:
            print(f"    Ref    : {refs[i - 1]}")
        print(f"    Greedy : {g_pred}")
        print(f"    Beam   : {b_pred}\n")


def _strip_tag(text: str) -> str:
    """'<2hi> how are you' -> 'how are you' (for display only)."""
    return text.split("> ", 1)[1] if text.startswith("<2") and "> " in text else text


def print_multilingual_samples(
    cfg: Any,
    generator: Any,
    eval_splits: dict[str, dict[str, list[str]]],
    target_langs: Iterable[str],
    valid_langs: Iterable[str],
    token_fmt: str,
    beam_size: int = 5,
) -> None:
    """Multilingual model.

    "test"   -> `num_samples` seeded random test pairs for EACH direction (en-hi, hi-en, ...).
    "config" -> every entry of sample_sentences is either "lang: text" (translate into that language) or
                plain English (translated into every target language of the training pairs).
    """
    valid = {v.lower() for v in valid_langs}
    entries: list[tuple[str, str, str, str | None]] = []  # (label, model input, shown source, reference)

    if cfg.inference.sample_source == "test":
        title = "Test"
        rng = random.Random(cfg.project.seed)
        for pair, split in eval_splits.items():
            n = len(split["test_src"])
            for i in rng.sample(range(n), min(cfg.inference.num_samples, n)):
                src = split["test_src"][i]
                entries.append((pair.upper().replace("-", " -> "), src, _strip_tag(src), split["test_tgt"][i]))
    else:
        title = "Config"
        for raw in cfg.inference.sample_sentences:
            head, sep, tail = raw.partition(":")
            if sep and head.strip().lower() in valid:
                targets, text = [head.strip().lower()], tail.strip()
            else:
                targets, text = sorted(target_langs), raw.strip()
            for tgt in targets:
                entries.append((f"-> {tgt.upper()}", f"{token_fmt.format(tgt)} {text}", text, None))

    print(f"\n=== {title} Sample Translations (multilingual) ===")
    if not entries:
        print("(no samples configured)\n")
        return
    sentences = [e[1] for e in entries]
    greedy_preds = generator.batched_greedy_decode(sentences)
    beam_preds = generator.batched_beam_decode(sentences, beam_size=beam_size)

    last_label = None
    for n, ((label, _, shown, ref), g_pred, b_pred) in enumerate(zip(entries, greedy_preds, beam_preds), 1):
        if label != last_label:
            print(f"--- {label} ---")
            last_label = label
        print(f"[{n}] SRC  : {shown}")
        if ref is not None:
            print(f"    Ref    : {ref}")
        print(f"    Greedy : {g_pred}")
        print(f"    Beam   : {b_pred}\n")
