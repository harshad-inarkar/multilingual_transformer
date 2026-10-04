from __future__ import annotations

from typing import Any, Iterator

import torch
import torch.nn.functional as F

from multilingual_transformer.data.dataset import numericalize_batch, pad_batch
from multilingual_transformer.models.transformer import MultilingualTransformer, make_src_mask, make_tgt_mask
from multilingual_transformer.utils.helpers import free_memory
from tqdm.auto import tqdm


def _banned_ngram_mask(tgt: torch.Tensor, n: int, vocab_size: int) -> torch.Tensor | None:
    length = tgt.size(1)
    if n < 2 or length < n:
        return None
    windows = tgt.unfold(1, n, 1)                      
    suffix = tgt[:, length - (n - 1):]                 
    match = (windows[:, :, :-1] == suffix.unsqueeze(1)).all(dim=-1)   
    rows, cols = match.nonzero(as_tuple=True)
    if rows.numel() == 0:
        return None
    banned = torch.zeros(tgt.size(0), vocab_size, dtype=torch.bool, device=tgt.device)
    banned[rows, windows[rows, cols, -1]] = True
    return banned


class TranslationGenerator:
    def __init__(
        self,
        model: MultilingualTransformer,
        src_tokenizer: Any,
        tgt_tokenizer: Any,
        max_len: int,
        device: torch.device,
        no_repeat_ngram_size: int = 0,
    ) -> None:
        self.no_repeat_ngram_size = no_repeat_ngram_size  
        self.src_tok = src_tokenizer
        self.tgt_tok = tgt_tokenizer
        self.max_len = max_len
        self.device = device
        self.pad_src = src_tokenizer.token_to_id("<pad>")
        self.pad_tgt = tgt_tokenizer.token_to_id("<pad>")
        self.sos_tgt = tgt_tokenizer.token_to_id("<sos>")
        self.eos_tgt = tgt_tokenizer.token_to_id("<eos>")
        self.model = model.to(device).eval()
        self._use_amp = device.type == "cuda"

    def _autocast(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=torch.float16, enabled=self._use_amp)

    def _to_text(self, ids: list[int]) -> str:
        if self.eos_tgt in ids:
            ids = ids[: ids.index(self.eos_tgt)]
        return self.tgt_tok.decode(ids, skip_special_tokens=True)

    def _batches(self, sentences: list[str], batch_size: int) -> Iterator[tuple[list[int], torch.Tensor, torch.Tensor]]:
        ids = numericalize_batch(sentences, self.src_tok, self.max_len)
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        for s in range(0, len(order), batch_size):
            idx = order[s : s + batch_size]
            src = pad_batch([ids[i] for i in idx], self.pad_src).to(self.device)
            yield idx, src, make_src_mask(src, self.pad_src)

    @staticmethod
    def _progress(batches, n_sentences: int, batch_size: int, desc: str):
        n_batches = -(-n_sentences // batch_size)
        return tqdm(batches, total=n_batches, desc=desc, leave=False, disable=n_batches <= 1)

    def _next_logits(self, tgt: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        tgt_mask = make_tgt_mask(tgt, self.pad_tgt)
        with self._autocast():
            logits = self.model.decoder(tgt, enc_out, src_mask, tgt_mask, last_only=True)
        return logits[:, -1, :].float()

    @torch.inference_mode()
    def greedy_decode(self, sentence: str, tgt_prefix_token: str | None = None) -> str:
        return self.batched_greedy_decode([sentence], batch_size=1, tgt_prefix_token=tgt_prefix_token)[0]

    @torch.inference_mode()
    def batched_greedy_decode(self, sentences: list[str], batch_size: int = 128, tgt_prefix_token: str | None = None) -> list[str]:
        results = [""] * len(sentences)
        for idx, src, src_mask in self._progress(
            self._batches(sentences, batch_size), len(sentences), batch_size, "Greedy decode"
        ):
            with self._autocast():
                enc_out = self.model.encoder(src, src_mask)

            if tgt_prefix_token:
                prefix_id = self.tgt_tok.token_to_id(tgt_prefix_token)
                tgt = torch.tensor([[self.sos_tgt, prefix_id]], dtype=torch.long, device=self.device).repeat(len(idx), 1)
            else:
                tgt = torch.full((len(idx), 1), self.sos_tgt, dtype=torch.long, device=self.device)

            unfinished = torch.ones(len(idx), dtype=torch.bool, device=self.device)
            for _ in range(self.max_len - 1):
                next_tokens = self._next_logits(tgt, enc_out, src_mask).argmax(dim=-1)
                tgt = torch.cat([tgt, next_tokens.unsqueeze(1)], dim=1)
                unfinished &= next_tokens != self.eos_tgt
                if not unfinished.any():
                    break

            for i, row in zip(idx, tgt.tolist()):
                results[i] = self._to_text(row)
            del enc_out, tgt, unfinished, src, src_mask
        free_memory()
        return results

    @torch.inference_mode()
    def batched_beam_decode(
        self,
        sentences: list[str],
        beam_size: int = 5,
        batch_size: int = 128,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int | None = None,
        tgt_prefix_token: str | None = None,
    ) -> list[str]:
        vocab_size = self.tgt_tok.get_vocab_size()
        n_block = self.no_repeat_ngram_size if no_repeat_ngram_size is None else no_repeat_ngram_size
        k = beam_size
        results = [""] * len(sentences)

        for idx, src, src_mask in self._progress(
            self._batches(sentences, batch_size), len(sentences), batch_size, f"Beam decode (k={k})"
        ):
            b = len(idx)
            with self._autocast():
                enc_out = self.model.encoder(src, src_mask)
            enc_out = enc_out.repeat_interleave(k, dim=0)
            src_mask = src_mask.repeat_interleave(k, dim=0)
            
            if tgt_prefix_token:
                prefix_id = self.tgt_tok.token_to_id(tgt_prefix_token)
                tgt = torch.tensor([[self.sos_tgt, prefix_id]], dtype=torch.long, device=self.device).repeat(b * k, 1)
            else:
                tgt = torch.full((b * k, 1), self.sos_tgt, dtype=torch.long, device=self.device)

            scores = torch.full((b, k), float("-inf"), device=self.device)
            scores[:, 0] = 0.0  
            finished = torch.zeros((b, k), dtype=torch.bool, device=self.device)
            row_offset = (torch.arange(b, device=self.device) * k).unsqueeze(1)

            for _ in range(self.max_len - 1):
                log_probs = F.log_softmax(self._next_logits(tgt, enc_out, src_mask), dim=-1).view(b, k, vocab_size)
                banned = _banned_ngram_mask(tgt, n_block, vocab_size)
                if banned is not None:
                    banned = banned.view(b, k, vocab_size) & ~finished.unsqueeze(-1)
                    log_probs = log_probs.masked_fill(banned, float("-inf"))
                log_probs = log_probs.masked_fill(finished.unsqueeze(-1), float("-inf"))
                eos_lp = log_probs[..., self.eos_tgt]
                log_probs[..., self.eos_tgt] = torch.where(finished, torch.zeros_like(eos_lp), eos_lp)

                cand = (scores.unsqueeze(-1) + log_probs).view(b, k * vocab_size)
                scores, top = cand.topk(k, dim=1)
                beam_idx = torch.div(top, vocab_size, rounding_mode="floor")
                token_idx = top % vocab_size

                parent = (beam_idx + row_offset).view(-1)
                tgt = torch.cat([tgt[parent], token_idx.view(-1, 1)], dim=1)
                finished = finished.gather(1, beam_idx) | (token_idx == self.eos_tgt)
                if finished.all():
                    break

            seq_len = tgt.size(1)
            seqs = tgt.view(b, k, seq_len)
            is_eos = seqs == self.eos_tgt
            first_eos = is_eos.long().argmax(dim=-1)
            lengths = torch.where(is_eos.any(-1), first_eos, torch.full_like(first_eos, seq_len - 1)).clamp(min=1)
            best = (scores / lengths.float().pow(length_penalty)).argmax(dim=1)
            best_seqs = seqs[torch.arange(b, device=self.device), best].tolist()

            for i, row in zip(idx, best_seqs):
                results[i] = self._to_text(row)
            del enc_out, tgt, seqs, scores, finished, src, src_mask
        free_memory()
        return results