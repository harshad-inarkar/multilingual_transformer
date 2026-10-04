from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>", "<unk>"]
_WORD_RE = re.compile(r"\w+|[^\w\s]")


class _Encoding(NamedTuple):
    ids: list[int]


def _split_text(text: str, split_type: str) -> list[str]:
    return _WORD_RE.findall(text) if split_type == "regex" else text.split()


class BasicVocabTokenizer:
    """Word-level tokenizer (whitespace / regex). Input is expected to be lowercased by the caller."""

    def __init__(self, vocab: dict[str, int], split_type: str = "whitespace") -> None:
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.split_type = split_type
        self._unk = vocab["<unk>"]

    def get_vocab(self) -> dict[str, int]:
        return self.vocab

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self._unk)

    def encode(self, text: str) -> _Encoding:
        return _Encoding([self.vocab.get(t, self._unk) for t in _split_text(text, self.split_type)])

    def encode_batch(self, texts: list[str]) -> list[_Encoding]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        specials = set(SPECIAL_TOKENS) if skip_special_tokens else set()
        tokens = (self.id_to_token.get(i, "<unk>") for i in ids)
        return " ".join(t for t in tokens if t not in specials)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "basic", "split_type": self.split_type, "vocab": self.vocab}


class TokenizerManager:
    @staticmethod
    def train(sentences: list[str], algo: str, vocab_size: int = 15000) -> Any:
        algo = algo.lower()
        # Lowercase lazily so train-time and encode-time text match (numericalize lowercases).
        corpus = (s.lower() for s in sentences)
        
        if algo in ("whitespace", "regex"):
            counter: Counter[str] = Counter()
            for text in corpus:
                counter.update(_split_text(text, algo))
            vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
            for token, _ in counter.most_common(vocab_size - len(SPECIAL_TOKENS)):
                if token not in vocab:
                    vocab[token] = len(vocab)
            return BasicVocabTokenizer(vocab, split_type=algo)
        
        trainer: Any
        if algo == "bpe":
            tok = Tokenizer(models.BPE(unk_token="<unk>"))
            tok.pre_tokenizer = pre_tokenizers.Sequence(
                [pre_tokenizers.Metaspace(), pre_tokenizers.Punctuation()]
            )
            tok.decoder = decoders.Metaspace()
            trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
        elif algo == "unigram":
            tok = Tokenizer(models.Unigram())
            tok.pre_tokenizer = pre_tokenizers.Sequence(
                [pre_tokenizers.Metaspace(), pre_tokenizers.Punctuation()]
            )
            tok.decoder = decoders.Metaspace()
            trainer = trainers.UnigramTrainer(
                vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, unk_token="<unk>"
            )
        elif algo == "wordpiece":
            tok = Tokenizer(models.WordPiece(unk_token="<unk>"))
            tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
            tok.decoder = decoders.WordPiece()
            trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
        else:
            raise ValueError(f"Unsupported tokenizer algorithm: {algo}")

        tok.train_from_iterator(corpus, trainer)
        return tok

    @staticmethod
    def save(tokenizer: Any, save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(tokenizer, Tokenizer):
            tokenizer.save(str(save_path))
        else:
            save_path.write_text(json.dumps(tokenizer.to_dict(), ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def load(load_path: Path) -> Any:
        text = Path(load_path).read_text(encoding="utf-8")
        data = json.loads(text)
        if data.get("type") == "basic":
            return BasicVocabTokenizer(data["vocab"], data["split_type"])
        return Tokenizer.from_str(text)
