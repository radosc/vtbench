"""DivPrune Hybrid — importance-weighted diversity extension of DivPrune.

A research variant of DivPrune that blends the paper's pure diversity
objective with an L2-norm-based importance signal. Controlled by alpha:

    score = alpha * importance + (1 - alpha) * diversity

where:
    importance = normalized L2 norm of each token (higher activation = keep)
    diversity  = normalized min cosine distance to already-selected tokens

alpha=0.0  → pure diversity (close to the paper's MMDP)
alpha=0.5  → equal weighting (default)
alpha=1.0  → pure importance ranking

Rationale: empirical testing on MMMU-Pro (technical/academic content)
showed that keeping high-activation tokens — which tend to encode text,
diagrams, and fine detail — buys ~0.5pp accuracy over pure MMDP. The
paper's purity argument holds for general scenes but leaves accuracy
on the table for technical benchmarks.

This is a research extension, NOT the paper's algorithm. Use the
`divprune` compressor for the paper-faithful implementation.
"""

import torch
import torch.nn.functional as F

from vtbench.compressors._base import Compressor


class DivPruneHybrid(Compressor):

    name = "divprune_hybrid"
    description = "DivPrune + L2-norm importance (research extension, alpha-weighted)"

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: balance between importance (1.0) and diversity (0.0).
                   Default 0.5 gives equal weight to both.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    def compress(self, features: torch.Tensor, n_target: int, **ctx) -> torch.Tensor:
        n = len(features)
        if n <= n_target:
            return features

        feat = features.float()
        device = features.device

        # Importance: L2 norm of each token embedding.
        norms = feat.norm(dim=1)
        imp_min, imp_max = norms.min(), norms.max()
        if imp_max > imp_min:
            importance = (norms - imp_min) / (imp_max - imp_min)
        else:
            importance = torch.ones(n, device=device)

        # Pairwise cosine similarity for diversity computation
        fn = F.normalize(feat, dim=1)
        sim = fn @ fn.T

        # Seed: most important token (hybrid uses importance, not farthest pair)
        selected = [importance.argmax().item()]

        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[selected[0]] = True

        # min_dist[i] = min cosine distance from token i to any selected token
        min_dist = (1.0 - sim[selected[0]]).clone()

        for _ in range(n_target - 1):
            # Normalize diversity to [0, 1] for fair combination with importance
            d_max = min_dist.max()
            diversity = min_dist / d_max if d_max > 0 else min_dist

            # Combined score, masked in one vectorized op
            score = self.alpha * importance + (1.0 - self.alpha) * diversity
            score[mask] = -1.0

            nx = score.argmax().item()
            selected.append(nx)
            mask[nx] = True

            # Update running min distances
            min_dist = torch.min(min_dist, 1.0 - sim[nx])

        return features[selected]
