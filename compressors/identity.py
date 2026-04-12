"""Identity — no-compression baseline.

Returns features unchanged (or uniformly subsampled if n_target < n).

Every benchmark should include this as a sanity check:
  - At ratio=1.0 it must produce identical results to stock inference.
  - At lower ratios it shows what naive uniform subsampling does,
    which is the floor that any real algorithm should beat.
"""

import torch

from vtbench.compressors._base import Compressor


class Identity(Compressor):

    name = "identity"
    description = "No compression baseline (uniform stride subsampling if ratio < 1)"

    def compress(self, features: torch.Tensor, n_target: int, **ctx) -> torch.Tensor:
        n = len(features)
        if n <= n_target:
            return features

        # Uniform stride subsampling — preserves spatial ordering.
        # Not random, so results are deterministic.
        # Evenly spaced indices including first and last token.
        indices = torch.round(
            torch.linspace(0, n - 1, n_target, device=features.device)
        ).long()
        return features[indices]
