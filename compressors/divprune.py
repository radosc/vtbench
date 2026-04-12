"""DivPrune — importance-weighted diversity token selection.

At each step, selects the token that maximizes:

    score = alpha * importance + (1 - alpha) * diversity

Where:
    importance = normalized L2 norm (higher activation = more informative)
    diversity  = min cosine distance to any already-selected token

Seed: the single most important (highest L2 norm) token.

This balances two objectives:
  - Keep tokens that carry the most visual information (high activation)
  - Keep tokens that are maximally different from each other (coverage)

alpha=0.5 (default) weights both equally.
alpha=1.0 is pure importance ranking (no diversity).
alpha=0.0 is pure FPS (equivalent to fps.py, but with normalization overhead).

Reference: developed for SAVT visual token compression, 2026.
"""

import torch
import torch.nn.functional as F

from vtbench.compressors._base import Compressor


class DivPrune(Compressor):

    name = "divprune"
    description = "Importance-weighted diversity selection (alpha balances L2-norm vs coverage)"

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: balance between importance (1.0) and diversity (0.0).
                   Default 0.5 gives equal weight to both.
        """
        self.alpha = alpha

    def compress(self, features: torch.Tensor, n_target: int, **ctx) -> torch.Tensor:
        n = len(features)
        if n <= n_target:
            return features

        feat = features.float()
        device = features.device

        # Importance: L2 norm of each token embedding.
        # Tokens with higher activation magnitude tend to encode more
        # visually salient content (objects, text, edges vs. flat sky).
        norms = feat.norm(dim=1)
        imp_min, imp_max = norms.min(), norms.max()
        if imp_max > imp_min:
            importance = (norms - imp_min) / (imp_max - imp_min)
        else:
            # All tokens have identical norms — importance is uninformative
            importance = torch.ones(n, device=device)

        # Pairwise cosine similarity for diversity computation
        fn = F.normalize(feat, dim=1)
        sim = fn @ fn.T

        # Seed: most important token
        selected = [importance.argmax().item()]

        # Boolean mask for O(1) exclusion instead of O(k) Python loop.
        # The old code did `for s in selected: score[s] = -1.0` which
        # issued k individual GPU scalar writes per iteration — quadratic
        # total and ~2x slower than FPS on real benchmarks.
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[selected[0]] = True

        # min_dist[i] = min cosine distance from token i to any selected token.
        # Starts with distance to the seed.
        min_dist = (1.0 - sim[selected[0]]).clone()

        for _ in range(n_target - 1):
            # Diversity: normalized min distance to selected set
            d_max = min_dist.max()
            diversity = min_dist / d_max if d_max > 0 else min_dist

            # Combined score, masked in one vectorized op
            score = self.alpha * importance + (1.0 - self.alpha) * diversity
            score[mask] = -1.0

            # Greedy select
            nx = score.argmax().item()
            selected.append(nx)
            mask[nx] = True

            # Update running min distances
            min_dist = torch.min(min_dist, 1.0 - sim[nx])

        return features[selected]
