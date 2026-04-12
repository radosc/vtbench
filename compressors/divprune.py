"""DivPrune — Diversity-based Visual Token Pruning.

Formulates token pruning as a Max-Min Diversity Problem (MMDP):
maximize the minimum pairwise cosine distance among selected tokens.
No importance heuristic, no L2 norm weighting — pure coverage of the
feature space.

Algorithm:
  1. Seed: find the pair (i, j) with maximum cosine distance, i.e. the
     two tokens that are furthest apart in the embedding space. This is
     the only initialization that doesn't bias selection toward any
     particular region.
  2. Iteratively add the token that maximizes the minimum distance to
     the already-selected set (greedy MMDP solver).

Computational complexity: O(N * M) greedy iterations over a precomputed
O(N^2) cosine similarity matrix.

Reference:
  Alvar et al. "DivPrune: Diversity-based Visual Token Pruning for
  Large Multimodal Models." arXiv:2503.02175 (2025).
"""

import torch
import torch.nn.functional as F

from vtbench.compressors._base import Compressor


class DivPrune(Compressor):

    name = "divprune"
    description = "Diversity-based visual token pruning (MMDP, Alvar et al. 2025)"

    def compress(self, features: torch.Tensor, n_target: int, **ctx) -> torch.Tensor:
        n = len(features)
        if n <= n_target:
            return features

        feat = features.float()
        device = features.device

        # Pairwise cosine similarity
        fn = F.normalize(feat, dim=1)
        sim = fn @ fn.T

        # Seed initialization: the pair of tokens with MAXIMUM cosine distance
        # (minimum similarity). This is the MMDP-correct starting point —
        # any other seed biases the greedy selection toward a particular
        # region of the embedding space.
        sim_for_seed = sim.clone()
        sim_for_seed.fill_diagonal_(float("inf"))
        flat_idx = sim_for_seed.argmin().item()
        i, j = flat_idx // n, flat_idx % n

        mask = torch.zeros(n, dtype=torch.bool, device=device)

        if n_target == 1:
            # Degenerate case: return one of the farthest pair
            mask[i] = True
            return features[[i]]

        selected = [i, j]
        mask[i] = True
        mask[j] = True

        # min_dist[k] = min cosine distance from token k to any selected token
        min_dist = torch.min(1.0 - sim[i], 1.0 - sim[j])

        # Greedy MMDP: at each step select the token whose minimum distance
        # to the already-selected set is largest.
        for _ in range(n_target - 2):
            min_dist[mask] = -1.0
            nx = min_dist.argmax().item()
            selected.append(nx)
            mask[nx] = True
            min_dist = torch.min(min_dist, 1.0 - sim[nx])

        return features[selected]
