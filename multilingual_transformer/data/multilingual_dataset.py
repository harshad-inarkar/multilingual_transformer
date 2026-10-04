from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from multilingual_transformer.data.sampler import BucketBatchSampler
import os

from multilingual_transformer.data.dataset import (
    PadCollator,
    TranslationDataset,
    is_clean_pair,
)


class MultilingualDataPipeline:
    def __init__(
        self,
        dataset_map: dict[str, str],
        data_dir: Path,
        max_len: int,
        max_ratio: float,
        seed: int,
    ) -> None:
        self.dataset_map = dataset_map
        self.data_dir = Path(data_dir)
        self.max_len = max_len
        self.max_ratio = max_ratio
        self.seed = seed

    def _extract_pair(self, row: dict[str, Any], ds_name: str, tgt_lang: str) -> tuple[str, str]:
        if ds_name == "acomquest/Saamayik":
            rec = row.get("translation", row)
            return str(rec.get("en", "")).strip(), str(rec.get("sa", "")).strip()
        return str(row["src"]).strip(), str(row["tgt"]).strip()

    def acquire_multilingual_corpus(
        self, pairs_per_lang: int, 
        val_per_lang: int, 
        test_per_lang: int, 
        force_download: bool = False, 
        verbose: bool = True, 
    ) -> dict[str, Any]:
        import pickle
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        lang_hash = "_".join(sorted(self.dataset_map.keys()))
        cache_name = (f"multi_corpus_{lang_hash}_{pairs_per_lang}_{val_per_lang}_{test_per_lang}"
              f"_s{self.seed}_l{self.max_len}_r{self.max_ratio}.pkl")
              
        cache_path = self.data_dir / cache_name
        
        if not force_download and cache_path.exists() and verbose:
            print(f"Loading cached multilingual corpus from {cache_path.name}...")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

        train_src, train_tgt = [], []
        eval_splits: dict[str, dict[str, list[str]]] = {}

        per_lang = {}

        for tgt_lang, ds_name in self.dataset_map.items():
            print(f"Loading {tgt_lang.upper()} pairs from {ds_name}...")
            total_needed = pairs_per_lang + val_per_lang + test_per_lang

            args = (ds_name, tgt_lang) if ds_name != "acomquest/Saamayik" else (ds_name,)
            try:
                ds = load_dataset(*args, split="train")
            except Exception:
                raise ValueError(f"Error load dataset {args}")

            ds = ds.shuffle(seed=self.seed)

            cur_en, cur_tgt = [], []
            seen = set()
            with tqdm(total=total_needed, desc=f"Extracting {tgt_lang.upper()}") as pbar:
                for row in ds:
                    en, tg = self._extract_pair(row, ds_name, tgt_lang)
                    key = en.lower()
                    if key in seen or not is_clean_pair(en, tg, self.max_len, self.max_ratio):
                        continue
                    seen.add(key)
                    cur_en.append(en)
                    cur_tgt.append(tg)
                    pbar.update(1)
                    if len(cur_en) >= total_needed:
                        break

            total_requested = pairs_per_lang + val_per_lang + test_per_lang
            actual_total = len(cur_en)
            
            bounds, cum = [0], 0
            for n in [pairs_per_lang, val_per_lang, test_per_lang]:
                cum += n
                bounds.append(int(actual_total * cum / total_requested))
                
            tr_e, tr_t = cur_en[bounds[0]:bounds[1]], cur_tgt[bounds[0]:bounds[1]]
            va_e, va_t = cur_en[bounds[1]:bounds[2]], cur_tgt[bounds[1]:bounds[2]]
            te_e, te_t = cur_en[bounds[2]:bounds[3]], cur_tgt[bounds[2]:bounds[3]]

            per_lang[tgt_lang] = (tr_e, tr_t, va_e, va_t, te_e, te_t)

        eval_en = {s.lower() for v in per_lang.values() for s in v[2] + v[4]}
        n_dropped = 0
        for tgt_lang, (tr_e, tr_t, va_e, va_t, te_e, te_t) in per_lang.items():
            kept = [(e, t) for e, t in zip(tr_e, tr_t) if e.lower() not in eval_en]
            n_dropped += len(tr_e) - len(kept)
            tr_e, tr_t = [e for e, _ in kept], [t for _, t in kept]

            # Source gets plain text, Target gets the target prefix token
            train_src.extend(tr_e)
            train_tgt.extend([f"<2{tgt_lang}> {t}" for t in tr_t])
            
            train_src.extend(tr_t)
            train_tgt.extend([f"<2en> {e}" for e in tr_e])
            
            eval_splits[f"en-{tgt_lang}"] = {
                "val_src": va_e, "val_tgt": [f"<2{tgt_lang}> {t}" for t in va_t],
                "test_src": te_e, "test_tgt": [f"<2{tgt_lang}> {t}" for t in te_t],
                "tgt_prefix": f"<2{tgt_lang}>"
            }
            eval_splits[f"{tgt_lang}-en"] = {
                "val_src": va_t, "val_tgt": [f"<2en> {e}" for e in va_e],
                "test_src": te_t, "test_tgt": [f"<2en> {e}" for e in te_e],
                "tgt_prefix": "<2en>"
            }
        if verbose:
            print(f"Removed {n_dropped:,} training pairs whose English text appears in a val/test split.")

        combined_val_src, combined_val_tgt = [], []
        for pair_data in eval_splits.values():
            combined_val_src.extend(pair_data["val_src"])
            combined_val_tgt.extend(pair_data["val_tgt"])

        out_dict = {
            "train_src": train_src,
            "train_tgt": train_tgt,
            "val_src": combined_val_src,
            "val_tgt": combined_val_tgt,
            "eval_splits": eval_splits,
        }
        
        tmp = cache_path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(out_dict, f)
        os.replace(tmp, cache_path)

        return out_dict

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