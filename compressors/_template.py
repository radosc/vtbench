"""Template — copy this file to create a new compressor.

Instructions:
  1. Copy this file: cp _template.py my_algorithm.py
  2. Rename the class and set `name` to your algorithm's identifier
  3. Implement compress()
  4. Done — `python -m vtbench list` will show your algorithm

The framework auto-discovers any .py file in this folder (vtbench/compressors/)
that contains a Compressor subclass. Files starting with underscore are skipped.

You can also keep your file outside this folder and load it via:
  python -m vtbench run --compressor /path/to/my_algorithm.py ...
"""

import torch

from vtbench.compressors._base import Compressor


class MyCompressor(Compressor):

    # This name is used in CLI (--compressor my_compressor) and result tables.
    # Must be unique across all compressors.
    # IMPORTANT: set a unique name here. Empty string prevents accidental
    # discovery if this template is copied without renaming.
    name = ""

    # One-line description shown in `python -m vtbench list`.
    description = "Brief description of what your algorithm does"

    def __init__(self, my_param: float = 0.5):
        """Set any hyperparameters for your algorithm here."""
        self.my_param = my_param

    def compress(self, features: torch.Tensor, n_target: int, **ctx) -> torch.Tensor:
        """Compress N vision tokens down to n_target.

        Args:
            features: [N, D] tensor — vision token embeddings (float).
                      N is typically 260 for Gemma 4 at standard resolution.
                      D is 1152 (SigLIP hidden dim for Gemma 4).
            n_target: how many tokens to output.
            **ctx:    optional extras. Safe to ignore.

        Returns:
            [M, D] tensor where M == n_target.

        Tips:
          - features is always float (promoted by pipeline before calling)
          - For selection-based methods: return features[selected_indices]
          - For merging methods: return new [M, D] tensor of merged features
          - Use F.normalize + cosine similarity for distance computations
          - features.norm(dim=1) gives L2 norms (useful as importance signal)
        """
        n = len(features)
        if n <= n_target:
            return features

        # ---- Replace this with your algorithm ----

        # Example: random selection (don't ship this)
        indices = torch.randperm(n, device=features.device)[:n_target]
        return features[indices]
