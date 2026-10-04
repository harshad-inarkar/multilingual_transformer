from __future__ import annotations

from collections import Counter
from typing import Any

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Import everything needed from your original untouched script
from multilingual_transformer.data.tokenizer import (
    SPECIAL_TOKENS,
    BasicVocabTokenizer,
    TokenizerManager,
    _split_text,
)

class MultilingualTokenizerManager(TokenizerManager):
    @staticmethod
    def train(
        sentences: list[str], 
        algo: str, 
        vocab_size: int = 15000, 
        lang_tokens: list[str] | None = None
    ) -> Any:
        algo = algo.lower()
        corpus = (s.lower() for s in sentences)
        
        # Combine base specials with our new multilingual tags
        custom_specials = SPECIAL_TOKENS + (lang_tokens or [])

        if algo in ("whitespace", "regex"):
            counter: Counter[str] = Counter()
            for text in corpus:
                counter.update(_split_text(text, algo))
            vocab = {tok: i for i, tok in enumerate(custom_specials)}
            for token, _ in counter.most_common(vocab_size - len(custom_specials)):
                if token not in vocab:
                    vocab[token] = len(vocab)
            return BasicVocabTokenizer(vocab, split_type=algo)

        trainer: Any
        if algo == "bpe":
            tok = Tokenizer(models.BPE(unk_token="<unk>"))
            tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Metaspace(), pre_tokenizers.Punctuation()])
            tok.decoder = decoders.Metaspace()
            trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=custom_specials)
            
        elif algo == "unigram":
            tok = Tokenizer(models.Unigram())
            tok.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Metaspace(), pre_tokenizers.Punctuation()])
            tok.decoder = decoders.Metaspace()
            trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=custom_specials, unk_token="<unk>")
            
        elif algo == "wordpiece":
            tok = Tokenizer(models.WordPiece(unk_token="<unk>"))
            tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
            tok.decoder = decoders.WordPiece()
            trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=custom_specials)
            
        else:
            raise ValueError(f"Unsupported tokenizer algorithm: {algo}")

        tok.train_from_iterator(corpus, trainer)
        return tok