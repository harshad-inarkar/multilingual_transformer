from __future__ import annotations

import unicodedata
from typing import TYPE_CHECKING

from sacrebleu.metrics import BLEU, CHRF
from multilingual_transformer.utils.distributed import gather_all

if TYPE_CHECKING:
    from multilingual_transformer.engine.decoder import TranslationGenerator


class TranslationEvaluator:
    def __init__(
        self,
        generator: TranslationGenerator,
        bleu_tokenize: str = "intl",
        chrf_word_order: int = 0,
        lowercase: bool = True,
        normalize_unicode: bool = True,
    ) -> None:
        self.generator = generator
        self.bleu_tokenize = bleu_tokenize
        self.chrf_word_order = chrf_word_order
        self.lowercase = lowercase
        self.normalize_unicode = normalize_unicode
        self.signatures: dict[str, str] = {}

    def _prep(self, text: str) -> str:
        return unicodedata.normalize("NFC", text) if self.normalize_unicode else text

    def evaluate(
        self,
        sources: list[str],
        references: list[str],
        method: str = "beam",
        beam_size: int = 5,
        batch_size: int = 128,
        sample_size: int = 500,
        rank: int = 0,
        world_size: int = 1,
        tgt_prefix_token: str | None = None,
    ) -> tuple[float, float, list[dict[str, str]]]:
        n = min(sample_size, len(sources))
        src_subset, ref_subset = sources[:n], references[:n]

        shard_src = [src_subset[i] for i in range(rank, len(src_subset), world_size)]
        shard_indices = list(range(rank, len(src_subset), world_size))

        if method == "beam":
            shard_preds = self.generator.batched_beam_decode(
                shard_src, beam_size=beam_size, batch_size=batch_size, tgt_prefix_token=tgt_prefix_token
            )
        else:
            shard_preds = self.generator.batched_greedy_decode(
                shard_src, batch_size=batch_size, tgt_prefix_token=tgt_prefix_token
            )

        gathered_data = gather_all(list(zip(shard_indices, shard_preds)), world_size)
        
        all_preds_dict: dict[int, str] = {}
        for rank_records in gathered_data:
            for idx, p in rank_records:
                all_preds_dict[idx] = p
        preds = [all_preds_dict[i] for i in range(len(src_subset))]

        hyps = [self._prep(p) for p in preds]
        refs = [self._prep(r) for r in ref_subset]

        bleu_metric = BLEU(tokenize=self.bleu_tokenize, lowercase=self.lowercase)
        chrf_metric = CHRF(word_order=self.chrf_word_order, lowercase=self.lowercase)
        bleu = bleu_metric.corpus_score(hyps, [refs]).score
        chrf = chrf_metric.corpus_score(hyps, [refs]).score
        self.signatures = {
            "bleu": str(bleu_metric.get_signature()),
            "chrf": str(chrf_metric.get_signature()),
        }

        samples = [{"src": src_subset[i], "ref": ref_subset[i], "pred": preds[i]} for i in range(min(5, n))]
        return float(bleu), float(chrf), samples