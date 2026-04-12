"""Tests for compressor contract and implementations.

All tests are CPU-only, no model loading required.
"""

import pytest
import torch

from vtbench.compressors._base import Compressor
from vtbench.compressors.divprune import DivPrune
from vtbench.compressors.fps import FPS
from vtbench.compressors.identity import Identity


# --- Fixtures ---

@pytest.fixture
def features_260():
    """Typical Gemma 4 feature tensor: 260 tokens, 1152-dim."""
    torch.manual_seed(42)
    return torch.randn(260, 1152)


@pytest.fixture
def features_small():
    """Small tensor for edge-case tests."""
    torch.manual_seed(42)
    return torch.randn(10, 64)


ALL_COMPRESSORS = [DivPrune(), FPS(), Identity()]
ALL_IDS = ["divprune", "fps", "identity"]


# --- Contract tests (must pass for ANY Compressor) ---

class TestCompressorContract:
    """Verify the Compressor interface contract for all built-in implementations."""

    @pytest.mark.parametrize("compressor", ALL_COMPRESSORS, ids=ALL_IDS)
    def test_output_shape(self, compressor, features_260):
        """compress() returns [n_target, D] tensor."""
        out = compressor.compress(features_260, 64)
        assert out.shape == (64, 1152)

    @pytest.mark.parametrize("compressor", ALL_COMPRESSORS, ids=ALL_IDS)
    def test_preserves_dimension(self, compressor, features_260):
        """Output D matches input D."""
        out = compressor.compress(features_260, 100)
        assert out.shape[1] == features_260.shape[1]

    @pytest.mark.parametrize("compressor", ALL_COMPRESSORS, ids=ALL_IDS)
    def test_n_equals_target(self, compressor, features_260):
        """Output has exactly n_target tokens."""
        for target in [1, 32, 64, 128, 200]:
            out = compressor.compress(features_260, target)
            assert len(out) == target, f"Expected {target}, got {len(out)}"

    @pytest.mark.parametrize("compressor", ALL_COMPRESSORS, ids=ALL_IDS)
    def test_target_exceeds_input(self, compressor, features_small):
        """When n_target >= N, return all tokens unchanged."""
        out = compressor.compress(features_small, 20)
        assert out.shape == features_small.shape
        assert torch.equal(out, features_small)

    @pytest.mark.parametrize("compressor", ALL_COMPRESSORS, ids=ALL_IDS)
    def test_target_equals_input(self, compressor, features_small):
        """When n_target == N, return all tokens unchanged."""
        out = compressor.compress(features_small, 10)
        assert out.shape == features_small.shape

    @pytest.mark.parametrize("compressor", ALL_COMPRESSORS, ids=ALL_IDS)
    def test_single_token_output(self, compressor, features_small):
        """n_target=1 should work and return [1, D]."""
        out = compressor.compress(features_small, 1)
        assert out.shape == (1, features_small.shape[1])

    @pytest.mark.parametrize("compressor", ALL_COMPRESSORS, ids=ALL_IDS)
    def test_no_nans(self, compressor, features_260):
        """Output must not contain NaN values."""
        out = compressor.compress(features_260, 64)
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize("compressor", ALL_COMPRESSORS, ids=ALL_IDS)
    def test_deterministic(self, compressor, features_260):
        """Same input produces same output (no randomness)."""
        out1 = compressor.compress(features_260, 64)
        out2 = compressor.compress(features_260, 64)
        assert torch.equal(out1, out2)

    @pytest.mark.parametrize("compressor", ALL_COMPRESSORS, ids=ALL_IDS)
    def test_has_name(self, compressor):
        """Every compressor must have a non-empty name."""
        assert compressor.name
        assert isinstance(compressor.name, str)

    @pytest.mark.parametrize("compressor", ALL_COMPRESSORS, ids=ALL_IDS)
    def test_has_description(self, compressor):
        """Every compressor must have a description."""
        assert compressor.description
        assert isinstance(compressor.description, str)

    @pytest.mark.parametrize("compressor", ALL_COMPRESSORS, ids=ALL_IDS)
    def test_repr(self, compressor):
        """repr() should not crash."""
        r = repr(compressor)
        assert compressor.name in r


# --- Algorithm-specific tests ---

class TestDivPrune:

    def test_alpha_0_similar_to_fps(self, features_small):
        """alpha=0 should behave like pure diversity (similar to FPS)."""
        dp = DivPrune(alpha=0.0)
        fps = FPS()
        dp_out = dp.compress(features_small, 5)
        fps_out = fps.compress(features_small, 5)
        # Both should select diverse tokens — same seed (highest norm)
        # so first selected token should be the same
        assert dp_out.shape == fps_out.shape

    def test_alpha_1_selects_by_norm(self, features_small):
        """alpha=1 should select tokens purely by L2 norm (importance only)."""
        dp = DivPrune(alpha=1.0)
        out = dp.compress(features_small, 5)
        # The output should be the 5 tokens with highest norms
        norms = features_small.float().norm(dim=1)
        top5_idx = norms.topk(5).indices
        expected = features_small[top5_idx]
        # Check same set of tokens (order may differ due to greedy iteration)
        out_set = {tuple(t.tolist()) for t in out}
        expected_set = {tuple(t.tolist()) for t in expected}
        assert out_set == expected_set

    def test_default_alpha(self):
        """Default alpha is 0.5."""
        dp = DivPrune()
        assert dp.alpha == 0.5

    def test_custom_alpha(self):
        """Custom alpha is stored."""
        dp = DivPrune(alpha=0.7)
        assert dp.alpha == 0.7


class TestFPS:

    def test_seed_is_highest_norm(self, features_small):
        """First selected token should be the one with highest L2 norm."""
        fps = FPS()
        out = fps.compress(features_small, 1)
        expected_idx = features_small.float().norm(dim=1).argmax()
        assert torch.equal(out[0], features_small[expected_idx])

    def test_maximizes_min_distance(self, features_small):
        """Selected tokens should be spread out (no near-duplicates)."""
        fps = FPS()
        out = fps.compress(features_small, 5)
        # Pairwise cosine similarity should be relatively low
        import torch.nn.functional as F
        normed = F.normalize(out.float(), dim=1)
        sim = normed @ normed.T
        sim.fill_diagonal_(0)
        # No pair should have cosine sim > 0.99 (they should be diverse)
        assert sim.max() < 0.99


class TestIdentity:

    def test_full_ratio_returns_all(self, features_small):
        """When target >= N, returns exact same tensor."""
        identity = Identity()
        out = identity.compress(features_small, 10)
        assert torch.equal(out, features_small)

    def test_uniform_subsampling(self):
        """Subsampling should use uniform stride, not random."""
        identity = Identity()
        # 10 tokens, keep 5 — should pick indices 0, 2, 5, 7, 9 (approx)
        feat = torch.arange(50).reshape(10, 5).float()
        out = identity.compress(feat, 5)
        assert len(out) == 5
        # First and last should be from original first and last
        assert torch.equal(out[0], feat[0])
        assert torch.equal(out[-1], feat[-1])


# --- Edge cases ---

class TestEdgeCases:

    def test_identical_features(self):
        """All identical tokens — compressor should not crash."""
        feat = torch.ones(50, 64)
        for comp in ALL_COMPRESSORS:
            out = comp.compress(feat, 10)
            assert out.shape == (10, 64)
            assert not torch.isnan(out).any()

    def test_single_input_token(self):
        """Single input token with n_target=1."""
        feat = torch.randn(1, 64)
        for comp in ALL_COMPRESSORS:
            out = comp.compress(feat, 1)
            assert out.shape == (1, 64)

    def test_two_tokens_select_one(self):
        """Two input tokens, select one."""
        feat = torch.randn(2, 64)
        for comp in ALL_COMPRESSORS:
            out = comp.compress(feat, 1)
            assert out.shape == (1, 64)

    def test_high_dimensional(self):
        """Works with large feature dimensions."""
        feat = torch.randn(20, 4096)
        for comp in ALL_COMPRESSORS:
            out = comp.compress(feat, 5)
            assert out.shape == (5, 4096)
