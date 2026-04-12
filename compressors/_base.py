"""Compressor base class — the blueprint for all token compression algorithms.

To add your algorithm:
  1. Copy _template.py to a new file in this folder (e.g. my_algo.py)
  2. Subclass Compressor, set `name`, implement `compress()`
  3. Done. The framework auto-discovers any .py file in this folder
     that isn't prefixed with underscore.

The only hard contract is:
  - Input:  [N, D] float tensor of vision token embeddings
  - Output: [M, D] float tensor where M <= N

Whether you get there by selecting a subset (indexing) or by merging
tokens (weighted averaging, clustering, etc.) is up to you.
"""

from abc import ABC, abstractmethod
from torch import Tensor


class Compressor(ABC):
    """Abstract base for visual token compression algorithms."""

    # Short identifier used in CLI (--compressor <name>), result tables,
    # and output filenames. Must be unique across all compressors.
    name: str = ""

    # One-line description shown in `python -m vtbench list`.
    description: str = ""

    @abstractmethod
    def compress(self, features: Tensor, n_target: int, **ctx) -> Tensor:
        """Compress vision tokens.

        Args:
            features: [N, D] vision token embeddings. Always float-promoted
                      by the pipeline before calling (safe for math).
            n_target: desired number of output tokens. Your implementation
                      should return exactly this many when possible.
            **ctx:    optional context the pipeline may pass. Current keys:
                      - (none in v1, reserved for future spatial info, etc.)
                      Your algorithm can safely ignore **ctx.

        Returns:
            [M, D] tensor where M == n_target (ideally), M <= N.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
