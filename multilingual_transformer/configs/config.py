from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import tomllib


T = TypeVar("T")


def _build(cls: type[T], raw: dict[str, Any]) -> T:
    """Instantiate a dataclass from a dict, warning about (and ignoring) unknown keys."""
    valid = {f.name for f in fields(cls) if f.init}  # type: ignore[arg-type]
    unknown = sorted(set(raw) - valid)
    if unknown:
        warnings.warn(f"Ignoring unknown config keys for {cls.__name__}: {unknown}")
    return cls(**{k: v for k, v in raw.items() if k in valid})  # type: ignore[call-arg]


@dataclass
class ProjectConfig:
    name: str = "multilingual_transformer"
    seed: int = 42


@dataclass
class LanguageConfig:
    # User-facing keys
    src_lang: str = "English"
    target_lang: str = "Hindi"
    dataset_name: str = "ai4bharat/samanantar"
    # Derived from the [language_map] table (not set directly)
    src_code: str = field(init=False, default="en")
    tgt_lang: str = field(init=False, default="hi")  # target language *code*
    lang_pair: str = field(init=False, default="EN-HI")

    def resolve(self, codes: dict[str, str]) -> None:
        lookup = {k.lower(): v for k, v in codes.items()}
        for name in (self.src_lang, self.target_lang):
            if name.lower() not in lookup:
                raise ValueError(f"Unknown language '{name}'. Add it to [language_map]. Known: {', '.join(codes)}")
        self.src_code = lookup[self.src_lang.lower()]
        self.tgt_lang = lookup[self.target_lang.lower()]
        self.lang_pair = f"{self.src_code}-{self.tgt_lang}".upper()

    @property
    def tag(self) -> str:
        return self.lang_pair.lower().replace("-", "_")


@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    tokenizer_dir: Path = Path("tokenizers")
    train_size: int = 50000
    val_size: int = 5000
    test_size: int = 5000
    max_len: int = 80
    max_length_ratio: float = 6.0
    force_download: bool = False

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.tokenizer_dir = Path(self.tokenizer_dir)


@dataclass
class TokenizerConfig:
    algo_src: str = "wordpiece"
    algo_tgt: str = "unigram"
    max_vocab_size: int = 22400


@dataclass
class ModelConfig:
    d_model: int = 512
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = "swiglu"  # Default to SwiGLU if not specified in TOML

    def __post_init__(self) -> None:
        self.activation = self.activation.lower()
        valid = {"swiglu", "relu", "relusquared", "silu", "gelu"}
        if self.activation not in valid:
            raise ValueError(f"model.activation must be one of {valid}")


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 224
    lr: float = 5.0e-4
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    torch_compile: bool = False
    gen_batch_size: int = 128
    bleu_sample: int = 300
    save_best: bool = False  # True: keep best-val checkpoint; False: keep latest


@dataclass
class InferenceConfig:
    sample_sentences: list[str] = field(default_factory=list)
    sample_source: str = "config"  # "config" = sample_sentences above, "test" = first N test pairs
    num_samples: int = 5  # used when sample_source = "test"
    no_repeat_ngram_size: int = 3  # beam search: forbid repeating any n-gram of this size (0 = off)

    def __post_init__(self) -> None:
        self.sample_source = self.sample_source.lower()
        if self.sample_source not in ("config", "test"):
            raise ValueError("inference.sample_source must be 'config' or 'test'")
        if self.no_repeat_ngram_size < 0:
            raise ValueError("inference.no_repeat_ngram_size must be >= 0")


@dataclass
class AppConfig:
    project: ProjectConfig
    language: LanguageConfig
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig

    @classmethod
    def from_toml(cls, path: str | Path) -> AppConfig:
        path = Path(path)
        with path.open("rb") as f:
            raw = tomllib.load(f)

        # Language name -> code registry lives in the same file, under [language_map].
        codes = raw.get("language_map", {})
        if not codes:
            raise ValueError(
                f"No [language_map] table found in {path}. "
                'Add one entry per language, e.g.  Hindi = "hi".'
            )

        language = _build(LanguageConfig, raw.get("language", {}))
        language.resolve(codes)

        return cls(
            project=_build(ProjectConfig, raw.get("project", {})),
            language=language,
            data=_build(DataConfig, raw.get("data", {})),
            tokenizer=_build(TokenizerConfig, raw.get("tokenizer", {})),
            model=_build(ModelConfig, raw.get("model", {})),
            training=_build(TrainingConfig, raw.get("training", {})),
            inference=_build(InferenceConfig, raw.get("inference", {})),
        )

    # ---- shared artifact paths (single source of truth) ----
    def checkpoint_path(self, kind: str | None = None) -> Path:
        kind = kind or ("best" if self.training.save_best else "last")
        return self.data.checkpoint_dir / f"transformer_{self.language.tag}_{kind}.pt"

    def tokenizer_path(self, side: str) -> Path:
        algo = self.tokenizer.algo_src if side == "src" else self.tokenizer.algo_tgt
        name = f"{self.language.tag}_{side}_{algo}_{self.tokenizer.max_vocab_size}.json"
        return self.data.tokenizer_dir / name