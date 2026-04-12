"""FPS — Farthest Point Sampling (Gonzalez, 1985).

Pure diversity-driven greedy selection in cosine distance space.

At each step, selects the token that is farthest (in cosine distance)
from all previously selected tokens:

    next = argmax_i  min_j(1 - cos_sim(token_i, selected_j))

Seed: the token with highest L2 norm (most activated).

Similar to DivPrune in objective but differs in seed initialization:
FPS starts from the highest L2 norm token, while DivPrune (MMDP)
starts from the pair with maximum pairwise distance. Both guarantee
that selected tokens maximize minimum pairwise distance.

Computational complexity: O(N * M) where N = input tokens, M = output tokens.
The cosine similarity matrix is O(N^2) but computed once upfront.

Reference: Gonzalez, T.F. "Clustering to minimize the maximum intercluster
distance." Theoretical Computer Science 38 (1985): 293-306.
"""

import torch
import torch.nn.functional as F

from vtbench.compressors._base import Compressor


class FPS(Compressor):

    name = "fps"
    description = "Farthest Point Sampling — greedy max-min cosine distance (Gonzalez 1985)"

    def compress(self, features: torch.Tensor, n_target: int, **ctx) -> torch.Tensor:
        n = len(features)
        if n <= n_target:
            return features

        feat = features.float()

        # Pairwise cosine similarity
        fn = F.normalize(feat, dim=1)
        sim = fn @ fn.T

        # Seed: highest L2 norm token
        selected = [feat.norm(dim=1).argmax().item()]

        # Mask prevents re-selecting an already-chosen token.
        # Without this, tied min_dist values could cause argmax to
        # return a previously selected index, producing duplicates.
        mask = torch.zeros(n, dtype=torch.bool, device=features.device)
        mask[selected[0]] = True

        # min_dist[i] = min cosine distance from token i to any selected token
        min_dist = (1.0 - sim[selected[0]]).clone()

        for _ in range(n_target - 1):
            min_dist[mask] = -1.0
            nx = min_dist.argmax().item()
            selected.append(nx)
            mask[nx] = True

            # Update running minimum distances
            min_dist = torch.min(min_dist, 1.0 - sim[nx])

        return features[selected]
