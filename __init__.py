"""SAVT — Visual Token Compression Benchmark for Vision-Language Models.

Drop-in framework for benchmarking token compression algorithms.
  - Add your algorithm: copy compressors/_template.py, implement compress()
  - Add a new model:    create models/<name>/, implement ModelBackend

Quick start:
    from vtbench import Pipeline
    from vtbench.compressors.divprune import DivPrune

    pipe = Pipeline("google/gemma-4-E4B-it")
    answer = pipe(image, "Describe this image.",
                  compressor=DivPrune(), ratio=0.5)

CLI:
    python -m vtbench list
    python -m vtbench run --model google/gemma-4-E4B-it --image photo.jpg
    python -m vtbench benchmark --model google/gemma-4-E4B-it --dataset gqa ...
"""

from vtbench.pipeline import Pipeline
from vtbench.compressors._base import Compressor
from vtbench.models._base import ModelBackend

__version__ = "0.1.0"
__all__ = ["Pipeline", "Compressor", "ModelBackend"]
