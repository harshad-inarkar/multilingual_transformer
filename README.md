# Multilingual Transformer

A from-scratch implementation of the Transformer (Vaswani et al., 2017) in PyTorch for **English ↔ Indic machine translation**. 

This pipeline supports both **One-to-One** translation (training a dedicated model for a specific language pair) and **Multilingual** translation (training a single model capable of translating across multiple languages using zero-shot target-forcing tokens).

## Key Architectural Features

- **Encoder-Only Target Forcing:** In multilingual mode, target language tags (e.g., `<2hi>`) are prepended to the encoder's input. Extensive benchmarking on ~26M parameter models demonstrates this yields significantly higher BLEU/chrF scores than decoder-side prompting or dual-forcing.
- **Automated Inference Pivot:** To bypass the "encoder entanglement" limitation in small models, the multilingual UI and interactive scripts utilize an automated English pivot. Zero-shot requests (e.g., `sa -> mr`) are seamlessly routed as `sa -> en -> mr` in the background, ensuring high-fidelity translation without requiring massive cross-lingual datasets.
- **Multi-GPU DistributedDataParallel (DDP):** Auto-detects GPUs and distributes batches via `mp.spawn`.
- **SwiGLU feed-forward:** gated FFN (SiLU gate × value, three weight matrices). With `d_ff = 3 × d_model`
- **Length-bucketed sampler:** sentences of similar length are batched together, removing most padding waste. The gain depends on your length distribution, so measure it on your data.
- **Bidirectional Prefix Augmentation:** Automatically trains the model on bidirectional translation in a single pass while rigorously filtering out English leakage across validation and test splits.

## Supported Languages and Datasets

| Language(s) | Dataset |
|---|---|
| Hindi | [`cfilt/iitb-english-hindi`](https://huggingface.co/datasets/cfilt/iitb-english-hindi) or Samanantar |
| Marathi, Bengali, Gujarati, Kannada, Malayalam, Odia, Punjabi, Tamil, Telugu, Assamese (and Hindi) | [`ai4bharat/samanantar`](https://huggingface.co/datasets/ai4bharat/samanantar) |
| Sanskrit | [`acomquest/Saamayik`](https://huggingface.co/datasets/acomquest/Saamayik) |

Datasets are downloaded automatically from Hugging Face via Arrow backends, pre-tokenized, and cached locally to bypass runtime overhead.

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA (T4 or better recommended; seamlessly supports 1 to *N* GPUs via DDP).
- PyTorch ≥ 2.3 ([install guide](https://pytorch.org/get-started/locally/))

## Installation

```bash
git clone https://github.com/harshad-inarkar/multilingual_transformer.git
cd multilingual_transformer
pip install -e .

## Quick Start (Terminal)

Run from a working directory; data, tokenizers, and checkpoints are written there.

### 1. One-to-One Translation (Single Language Pair)

```bash
mkdir -p work_dir && cd work_dir

# Auto-detects GPU count. If >1, it spawns DDP workers across all cards.
python -m multilingual_transformer.scripts.train          

# Interactive testing
python -m multilingual_transformer.scripts.interactive    

```

### 2. Universal Multilingual Model (Many-to-Many)

```bash
mkdir -p work_dir && cd work_dir

# Trains a single model on all language pairs defined in multilingual_config.toml
python -m multilingual_transformer.scripts.multilingual_train          

# Interactive testing with prefix target forcing (e.g., "mr: how are you?")
python -m multilingual_transformer.scripts.multilingual_interactive

```

## Jupyter & Colab Widgets

You can run interactive translation testing directly inside a notebook. The widget dynamically populates based on the configuration file used during training.

```bash
# For One-to-One Models
from multilingual_transformer.ui.widget import launch
launch()

# For Universal Multilingual Models
from multilingual_transformer.ui.multilingual_widget import launch
launch()

```
## Google Colab Setup:
1. Enable a GPU runtime: *Runtime → Change runtime type → T4 GPU*.
2. Clone and install:

```bash
# Cell 1: Setup, Install, and Create Working Directory
base_dir = '/content'
repo_dir = 'multilingual_transformer'
work_dir = f"{base_dir}/{repo_dir}/work_dir"

%cd {base_dir}

!rm -rf multilingual_transformer
!git clone https://github.com/harshad-inarkar/multilingual_transformer.git
%cd multilingual_transformer
!pip install -q -e .
!mkdir -p {work_dir}
```

3. Train and Interactive Script

```bash
# For One-to-One Models
%cd {work_dir}
!python -m multilingual_transformer.scripts.train

%cd {work_dir}
!python -m multilingual_transformer.scripts.interactive

# For Universal Multilingual Models
%cd {work_dir}
!python -m multilingual_transformer.scripts.multilingual_train

%cd {work_dir}
!python -m multilingual_transformer.scripts.multilingual_interactive

```

### Checkpoints, evaluation and GPUs

Both `*_best.pt` and `*_last.pt` are always saved. `save_best` only chooses which one evaluation, the REPL and the widget load.

```bash
python -m multilingual_transformer.scripts.evaluate --checkpoint best --sample-size 5000 --show-samples
```

Flags: `--method {both,greedy,beam}`, `--beam-size`, `--no-repeat-ngram`, `--checkpoint {best,last}`. This script is single-GPU and for one-to-one models only. Scoring is sacreBLEU (`tok:intl`, lowercase) plus chrF; the exact signatures are printed with every result. `batch_size` is per GPU, so two GPUs give twice the global batch. `torchrun --standalone --nproc_per_node=N -m ...` also works.



## Configuration

Configuration is managed via TOML files in `multilingual_transformer/configs/`.

### 1. `transformer_config.toml` (One-to-One)

```toml
[language]
src_lang = "English"
target_lang = "Hindi"
dataset_name = "ai4bharat/samanantar"

```

### 2. `multilingual_config.toml` (Universal Model)

The multilingual pipeline builds a shared vocabulary across all datasets and injects `<2tgt>` tags.

```toml
[multilingual]
train_languages_pairs = ["en-hi", "en-mr", "en-sa"]
target_tokens_format = "<2{}>"
pairs_per_lang = 50000

```

### Architecture Tuning

The architecture defaults to a highly efficient SwiGLU 3x dimension configuration. Batch sizes represent the load *per GPU*.

```toml
[model]
d_model = 256
num_layers = 4
num_heads = 4
d_ff = 768    # 3x d_model for optimal SwiGLU parameter density
dropout = 0.1

[training]
epochs = 10
batch_size = 256

```

## Project Structure

```
multilingual_transformer/
├── configs/     config.py, transformer_config.toml, multilingual_config.toml
├── data/        dataset.py, multilingual_dataset.py, sampler.py, tokenizer.py, multilingual_tokenizer.py
├── models/      attention.py, layers.py, transformer.py
├── engine/      trainer.py, decoder.py, evaluator.py, loader.py
├── scripts/     train.py, multilingual_train.py, interactive.py, multilingual_interactive.py, evaluate.py
├── ui/          widget.py, multilingual_widget.py
└── utils/       helpers.py, distributed.py, reporting.py

```

## References

* Shazeer, *[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)*, 2020
* Vaswani et al., *[Attention Is All You Need](https://arxiv.org/abs/1706.03762)*, 2017
* Ramesh et al., *[Samanantar](https://arxiv.org/abs/2104.05596)*, 2021
* Kunchukuttan et al., *[The IIT Bombay English-Hindi Parallel Corpus](https://arxiv.org/abs/1710.02855)*, 2018
* Maheshwari et al., *[Sāmayik](https://arxiv.org/abs/2305.14004)*, 2024

## License

The source code is released under the [MIT License](LICENSE).
