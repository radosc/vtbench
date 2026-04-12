# VTBench

**Visual Token Compression Benchmark for Vision-Language Models**

When a vision-language model like Gemma 4 processes an image, its vision encoder converts the image into a sequence of **vision tokens** — dense vector representations that the language model attends to alongside text tokens. Gemma 4 produces **~260 vision tokens per image** at standard resolution. Every one of these tokens consumes compute and memory during generation: they occupy KV-cache, participate in every attention layer, and scale quadratically with sequence length.

**Vision token compression** reduces this count while preserving as much visual information as possible:

| Configuration | Vision Tokens | Compression | What changes |
|---|---|---|---|
| Stock (no compression) | 260 | 1x | Baseline |
| 2x compression (ratio=0.5) | 130 | 2x | Half the vision tokens, ~same accuracy |
| 4x compression (ratio=0.25) | 65 | 4x | Quarter the vision tokens, some accuracy loss |

The question is: **which compression algorithm loses the least accuracy?** VTBench answers this. Drop in your algorithm, point it at a model, get a benchmark comparing it against baselines.

## Install

```bash
git clone <repo-url>
cd vtbench
pip install -e ".[all]"
```

This installs vtbench with all optional dependencies (MMMU-Pro dataset support, quantization, tests). For a minimal install use `pip install -e .` instead.

Requirements: Python 3.10+, CUDA GPU (8 GB+ VRAM for smallest model).

## Quick Start

```bash
# See available models, datasets, and compressors
python -m vtbench list

# Fetch a dataset (auto-downloads from HuggingFace)
python -m vtbench fetch mmmu_pro

# Benchmark two compressors at 2x and 4x compression
python -m vtbench benchmark \
  --model gemma-4-E4B-it \
  --data mmmu_pro \
  --compressors divprune fps \
  --ratios 0.5 0.25
```

The model downloads automatically on first use. Results print to terminal and save as JSON.

## What `vtbench list` Shows

```
Models:
  gemma-4-E2B-it         Gemma 4 E2B (2B params, smallest)           ~6 GB  [bf16, 8bit]
  gemma-4-E4B-it         Gemma 4 E4B (4B params, recommended)       ~18 GB  [bf16, 8bit]
  gemma-4-E12B-it        Gemma 4 E12B (12B params)                  ~42 GB  [bf16, 8bit]
  gemma-4-E27B-it        Gemma 4 E27B (27B params, largest)         ~65 GB  [bf16, 8bit]

Datasets:
  mmmu_pro         MMMU-Pro multiple choice                          [not fetched, ~1600 samples]
  gqa              GQA visual QA                                     [not fetched, ~132000 samples]

Compressors:
  divprune         Importance-weighted diversity selection
  fps              Farthest Point Sampling (Gonzalez 1985)
  identity         No compression baseline
```

Models, datasets, and compressors are all auto-discovered. Add more by dropping files in the right folders.

## Benchmark Output

```
========================================================================
  Model: gemma-4-E4B-it  |  Data: mmmu_pro (1592 samples)  |  Seed: 42
========================================================================
  Compressor           Ratio     Tokens   Accuracy   vs Stock
  ------------------------------------------------------------------
  stock                  1.00       260      37.3%         --
  divprune_0.50          0.50  260->130      35.7%     -1.6pp
  divprune_0.25          0.25   260->65      33.1%     -4.2pp
  fps_0.50               0.50  260->130      34.5%     -2.8pp
  fps_0.25               0.25   260->65      31.8%     -5.5pp
========================================================================

  Per-category breakdown (categories with >= 10 samples):
  Category              stock     divprune_0.50        fps_0.50
  ---------------------------------------------------------------
  Biology          45.2% ( 42)     43.1% ( 42)     41.0% ( 42)
  Chemistry        31.5% ( 89)     29.2% ( 89)     28.1% ( 89)
  Physics          38.0% ( 71)     37.0% ( 71)     35.2% ( 71)
```

The `Tokens` column shows exactly what's happening: 260 native tokens compressed to 130 (2x) or 65 (4x). `vs Stock` shows the accuracy cost. Results save as JSON with per-sample answers and per-category breakdowns. Benchmarks checkpoint every 10 samples and resume automatically.

## Fetching Datasets

```bash
# Auto-downloads from HuggingFace Hub
python -m vtbench fetch mmmu_pro

# GQA requires a local source (manual download from Stanford)
python -m vtbench fetch gqa --source /path/to/gqa_dir
```

Datasets are converted to a universal JSONL manifest and stored in `~/.vtbench/datasets/`.

### Custom Datasets

Create a JSONL file with one JSON object per line:

```json
{"id": "001", "image": "images/001.jpg", "prompt": "What is this?", "answer": "A cat", "category": "animals"}
{"id": "002", "image": "images/002.png", "prompt": "How many?", "answer": "3", "category": "counting"}
```

Image paths are relative to the manifest file. Then:

```bash
python -m vtbench benchmark --data ./my_manifest.jsonl --model gemma-4-E4B-it --compressors divprune
```

## Running Benchmarks

### CLI flags

```bash
python -m vtbench benchmark \
  --model gemma-4-E4B-it \
  --data mmmu_pro \
  --compressors divprune fps \
  --ratios 0.5 0.25 \
  --n 100 --seed 42
```

### Config file

```bash
python -m vtbench benchmark --config experiment.json
```

```json
{
    "model": "gemma-4-E4B-it",
    "data": "mmmu_pro",
    "compressors": ["divprune", "fps"],
    "ratios": [0.5, 0.25],
    "n_samples": 100,
    "seed": 42,
    "output_dir": "results/my_experiment",
    "gen_config": {"max_new_tokens": 10}
}
```

### Python API

```python
from vtbench import Pipeline
from vtbench.compressors.divprune import DivPrune

pipe = Pipeline("google/gemma-4-E4B-it")

# Stock: 260 vision tokens
answer = pipe(image, "What is in this image?")

# 2x compression: 260 → 130 vision tokens
answer = pipe(image, "What is in this image?",
              compressor=DivPrune(), ratio=0.5)

# 4x compression: 260 → 65 vision tokens
answer = pipe(image, "What is in this image?",
              compressor=DivPrune(), ratio=0.25)
```

### Single image

```bash
python -m vtbench run \
  --model gemma-4-E4B-it \
  --image photo.jpg \
  --prompt "Describe this image." \
  --compressor divprune --ratio 0.5
```

## Adding Your Algorithm

This is the main use case. You have a token compression algorithm and want to benchmark it against baselines.

1. Copy `vtbench/compressors/_template.py` to `vtbench/compressors/your_algo.py`
2. Set `name` and `description`
3. Implement `compress()` — it receives **N vision token embeddings** and must return **n_target** of them
4. Done. `vtbench list` shows it, and you can benchmark it immediately.

```python
import torch
from vtbench.compressors._base import Compressor

class MyAlgorithm(Compressor):
    name = "my_algo"
    description = "Brief description"

    def compress(self, features: torch.Tensor, n_target: int, **ctx) -> torch.Tensor:
        # features: [N, D] — N vision tokens, each a D-dimensional embedding
        #   N ≈ 260 for Gemma 4 at standard resolution
        #   D = 1152 (SigLIP hidden dimension)
        #
        # Return: [n_target, D] — your selected or merged tokens
        #
        # Two approaches:
        #   Selection: pick n_target tokens by index → features[indices]
        #   Merging:   combine tokens into n_target new ones → weighted averages
        ...
```

External files also work without modifying the package:

```bash
python -m vtbench benchmark \
  --model gemma-4-E4B-it \
  --data mmmu_pro \
  --compressors divprune fps /path/to/my_algo.py \
  --ratios 0.5 0.25
```

### Scoring

Auto-detected from the ground truth format in your dataset:
- **Single letter A-J**: letter extraction, robust to verbose output ("The answer is B")
- **Multi-word answer**: soft string match (GQA-style evaluation)

## Adding a Model Backend

To benchmark compression on a different VLM (not just Gemma 4):

1. Create `vtbench/models/your_model/`
2. Implement `ModelBackend` in `backend.py` — 5 methods: `supports`, `load`, `extract`, `generate_stock`, `generate_compressed`
3. Add a `MODELS` dict with available variants and quantization options
4. Export as `Backend` in `__init__.py`

See `vtbench/models/gemma4/` for a reference implementation with documented settings.

## Adding a Dataset

1. Create `vtbench/datasets/your_dataset.py`
2. Subclass `DatasetEntry`, implement `fetch()`
3. `fetch()` downloads data and produces a JSONL manifest
4. Done. `vtbench fetch your_dataset` works.

## Built-in Compressors

| Name | Tokens (2x) | Method | Reference |
|---|---|---|---|
| `divprune` | 260 → 130 | Selects tokens that maximize importance (L2 norm) + diversity (cosine distance). Tunable `alpha` parameter balances the two. | DivPrune: Diversity-based Visual Token Pruning for Large Multimodal Models, Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, Yong Zhang, 2025 |
| `fps` | 260 → 130 | Farthest Point Sampling — greedy max-min cosine distance. Pure spatial coverage, no importance weighting. | Gonzalez, 1985 |
| `identity` | 260 → 130 | Uniform stride subsampling. The floor any real algorithm should beat. | — |

## Architecture

```
vtbench/
├── pipeline.py              # Orchestrator: model + compressor
├── compressors/             # Drop a .py file = new algorithm
│   ├── _base.py             # Compressor ABC
│   ├── _template.py         # Copy this to start
│   ├── divprune.py          # Importance × diversity selection
│   ├── fps.py               # Farthest Point Sampling
│   └── identity.py          # No-op baseline
├── models/                  # Drop a folder = new model
│   ├── _base.py             # ModelBackend ABC
│   └── gemma4/              # Gemma 4 E2B/E4B/E12B/E27B
├── datasets/                # Drop a .py file = new dataset
│   ├── _base.py             # DatasetEntry ABC + JSONL loader
│   ├── mmmu_pro.py          # Auto-fetches from HuggingFace
│   └── gqa.py               # Converts from local download
├── benchmark/
│   ├── runner.py            # Sweep, checkpoint, resume, tables
│   └── scoring.py           # MC letter extraction + soft match
├── tests/                   # 111 tests, CPU-only
├── cli.py                   # list / fetch / run / benchmark
└── pyproject.toml           # pip install -e .
```

Three plugin axes, all auto-discovered from the filesystem:
- **Compressors**: `.py` in `compressors/` with a `Compressor` subclass
- **Model backends**: subfolder in `models/` with a `Backend` class
- **Datasets**: `.py` in `datasets/` with a `DatasetEntry` subclass

## Multi-GPU

Models are loaded with `device_map="auto"`, which distributes layers across all available GPUs automatically. Large models like E27B (65 GB) that don't fit on a single GPU work out of the box across multiple GPUs — no configuration needed.

## Gemma 4 Notes

The Gemma 4 backend (`vtbench/models/gemma4/backend.py`) documents settings validated across hundreds of benchmark runs:
- **`min_new_tokens=1`**: prevents ~33% empty-answer rate with greedy decoding
- **8-bit minimum quantization**: 4-bit causes severe hallucination on all vision tasks (8% accuracy vs 37% at bf16)
- **`bfloat16`**: native training precision, no benefit from float16/float32

## Tests

```bash
pip install -e ".[dev]"
python -m pytest vtbench/tests/ -v
```

111 tests, CPU-only, ~14 seconds. No GPU or model download needed.

## Troubleshooting

**`ModuleNotFoundError: No module named 'vtbench'`** — Run `pip install -e .` from inside the `vtbench/` directory.

**`ImportError: Gemma4VideoProcessor requires Torchvision`** — Run `pip install torchvision` (included automatically with `pip install -e ".[all]"`).

**`ImportError: bitsandbytes`** when using 8-bit/4-bit — Run `pip install bitsandbytes`. Linux/WSL only; bitsandbytes has limited Windows support.

**Empty model answers (~33% of images)** — Check that `min_new_tokens=1` is set in gen_config. The default Gemma 4 backend already handles this, but custom gen_config overrides may clear it.

**MMMU-Pro fetch fails** — Run `pip install datasets` or `pip install -e ".[mmmu]"`.

## License

Apache 2.0
