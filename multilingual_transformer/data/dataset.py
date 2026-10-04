from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from multilingual_transformer.data.sampler import BucketBatchSampler

DATASET_LANGS: dict[str, set[str]] = {
    "cfilt/iitb-english-hindi": {"hi"},
    "acomquest/Saamayik": {"sa"},
    "ai4bharat/samanantar": {"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"},
}


def is_clean_pair(en: str, tgt: str, max_len: int, max_ratio: float) -> bool:
    n_en, n_tgt = len(en.split()), len(tgt.split())
    if n_en < 3 or n_tgt < 3 or n_en > max_len - 2 or n_tgt > max_len - 2:
        return False
    return (max(n_en, n_tgt) / max(min(n_en, n_tgt), 1)) <= max_ratio


def numericalize_batch(texts: list[str], tokenizer: Any, max_len: int) -> list[list[int]]:
    sos_id = tokenizer.token_to_id("<sos>")
    eos_id = tokenizer.token_to_id("<eos>")
    encoded = tokenizer.encode_batch([t.lower() for t in texts])
    return [[sos_id] + e.ids[: max_len - 2] + [eos_id] for e in encoded]


def pad_batch(seqs: list[list[int]], pad_idx: int) -> Tensor:
    out = torch.full((len(seqs), max(len(s) for s in seqs)), pad_idx, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


class PadCollator:
    def __init__(self, pad_src: int, pad_tgt: int) -> None:
        self.pad_src = pad_src
        self.pad_tgt = pad_tgt

    def __call__(self, batch: list[tuple[list[int], list[int]]]) -> tuple[Tensor, Tensor]:
        src, tgt = zip(*batch)
        return pad_batch(list(src), self.pad_src), pad_batch(list(tgt), self.pad_tgt)


class TranslationDataset(Dataset):
    def __init__(
        self,
        src_texts: list[str],
        tgt_texts: list[str],
        src_tokenizer: Any,
        tgt_tokenizer: Any,
        max_len: int,
    ) -> None:
        self.src_ids = numericalize_batch(src_texts, src_tokenizer, max_len)
        self.tgt_ids = numericalize_batch(tgt_texts, tgt_tokenizer, max_len)
        self.pad_src = src_tokenizer.token_to_id("<pad>")
        self.pad_tgt = tgt_tokenizer.token_to_id("<pad>")

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int]]:
        return self.src_ids[idx], self.tgt_ids[idx]


class DataPipeline:
    def __init__(
        self,
        dataset_name: str,
        src_lang: str,
        tgt_lang: str,
        lang_pair: str,
        data_dir: Path,
        max_len: int,
        max_ratio: float,
        seed: int,
    ) -> None:
        if src_lang != "en":
            raise ValueError("All supported datasets have English as the source language.")
        supported = DATASET_LANGS.get(dataset_name)
        if supported is not None and tgt_lang not in supported:
            raise ValueError(
                f"Dataset '{dataset_name}' does not provide '{tgt_lang}'. Supported: {sorted(supported)}"
            )
        self.dataset_name = dataset_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.lang_pair = lang_pair
        self.data_dir = data_dir
        self.max_len = max_len
        self.max_ratio = max_ratio
        self.seed = seed

    def _cache_paths(self, sizes: dict[str, int]) -> dict[str, Path]:
        tag = self.lang_pair.lower().replace("-", "_")
        return {name: self.data_dir / f"{name}_{tag}_{n}.jsonl" for name, n in sizes.items()}

    def _open_dataset(self) -> Any:
        from datasets import load_dataset
        
        args: tuple[str, ...] = (self.dataset_name,)
        if self.dataset_name not in ("cfilt/iitb-english-hindi", "acomquest/Saamayik"):
            args = (self.dataset_name, self.tgt_lang)
        try:
            ds = load_dataset(*args, split="train")
        except (TypeError, ValueError, RuntimeError):
            raise ValueError(f"Error load dataset {args}")


        return ds.shuffle(seed=self.seed)

    def _extract_pair(self, row: dict[str, Any]) -> tuple[str, str]:
        if self.dataset_name == "cfilt/iitb-english-hindi":
            rec = row["translation"]
        elif self.dataset_name == "acomquest/Saamayik":
            rec = row.get("translation", row)
        else:
            return str(row["src"]).strip(), str(row["tgt"]).strip()
        return str(rec.get(self.src_lang, "")).strip(), str(rec.get(self.tgt_lang, "")).strip()

    def acquire_corpus(
        self, train_size: int, 
        val_size: int, 
        test_size: int, 
        force_download: bool = False,
        verbose: bool = False,

    ) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
        sizes = {"train": train_size, "val": val_size, "test": test_size}
        paths = self._cache_paths(sizes)
        if not force_download and all(p.exists() for p in paths.values()):
            if verbose:
                print(f"Loading cached dataset splits from {self.data_dir}...")

            out: list[list[str]] = []
            for p in paths.values():
                out.extend(self._load_jsonl(p))
            return tuple(out)

        total = sum(sizes.values())
        ds = self._open_dataset()

        src: list[str] = []
        tgt: list[str] = []
        seen: set[str] = set()
        print(f"Filtering {total:,} random clean pairs from the Arrow-backed dataset {self.dataset_name}...")

        with tqdm(total=total, desc="Extracting Pairs") as pbar:
            for row in ds:
                en, tg = self._extract_pair(row)
                key = en.lower()
                if key in seen or not is_clean_pair(en, tg, self.max_len, self.max_ratio):
                    continue
                seen.add(key)
                src.append(en)
                tgt.append(tg)
                pbar.update(1)
                if len(src) >= total:
                    break
        del seen, ds

        bounds, cum = [0], 0
        for n in sizes.values():
            cum += n
            bounds.append(int(len(src) * cum / total))
        out = []
        for name, lo, hi in zip(sizes, bounds[:-1], bounds[1:]):
            s_part, t_part = src[lo:hi], tgt[lo:hi]
            self._save_jsonl(paths[name], s_part, t_part)
            out.extend([s_part, t_part])
        del src, tgt
        return tuple(out)

    def load_cached_split(self, name: str, size: int) -> tuple[list[str], list[str]]:
        path = self._cache_paths({name: size})[name]
        if not path.exists():
            raise FileNotFoundError(f"Cached {name} split not found at {path}. Run training first.")
        return self._load_jsonl(path)

    @staticmethod
    def _save_jsonl(path: Path, src: list[str], tgt: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for s, t in zip(src, tgt):
                fh.write(json.dumps({"src": s, "tgt": t}, ensure_ascii=False) + "\n")

    @staticmethod
    def _load_jsonl(path: Path) -> tuple[list[str], list[str]]:
        src, tgt = [], []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    item = json.loads(line)
                    src.append(item.get("src", item.get("en")))
                    tgt.append(item["tgt"])
        return src, tgt

    @staticmethod
    def create_loader(
        dataset: TranslationDataset,
        batch_size: int,
        shuffle: bool,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ) -> DataLoader:
        lengths = [max(len(s), len(t)) for s, t in zip(dataset.src_ids, dataset.tgt_ids)]
        sampler = BucketBatchSampler(
            lengths, batch_size, shuffle=shuffle, seed=seed, rank=rank, world_size=world_size
        )

        return DataLoader(
            dataset,
            batch_sampler=sampler,
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
            collate_fn=PadCollator(dataset.pad_src, dataset.pad_tgt),
        )