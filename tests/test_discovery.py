"""Tests for auto-discovery of compressors and model backends.

CPU-only, no model loading required.
"""

import pytest
import tempfile
import os
from pathlib import Path

from vtbench.compressors._base import Compressor
from vtbench.compressors._discover import (
    discover_compressors,
    load_external_compressor,
    list_compressors,
)
from vtbench.models._discover import discover_backend, list_backends
from vtbench.models._base import ModelBackend


class TestCompressorDiscovery:

    def test_discovers_builtins(self):
        """All three built-in compressors should be discovered."""
        comps = discover_compressors()
        assert "divprune" in comps
        assert "fps" in comps
        assert "identity" in comps

    def test_discovered_are_compressor_subclasses(self):
        """Every discovered class must be a Compressor subclass."""
        for name, cls in discover_compressors().items():
            assert issubclass(cls, Compressor)
            assert cls is not Compressor

    def test_list_compressors(self):
        """list_compressors returns sorted names."""
        names = list_compressors()
        assert names == sorted(names)
        assert "divprune" in names

    def test_underscore_files_skipped(self):
        """Files starting with _ should not be discovered."""
        comps = discover_compressors()
        # _template.py is underscore-prefixed (skipped by discovery) AND has
        # name="" (filtered by the `and obj.name` check). Double protection.
        assert "" not in comps
        assert all(name for name in comps)  # no empty names

    def test_load_external_compressor(self):
        """Loading a compressor from an external .py file."""
        code = '''
import torch
from vtbench.compressors._base import Compressor

class TestExternal(Compressor):
    name = "test_external"
    description = "test"
    def compress(self, features, n_target, **ctx):
        return features[:n_target]
'''
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="."
        ) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        try:
            cls = load_external_compressor(tmp_path)
            assert cls.name == "test_external"
            assert issubclass(cls, Compressor)

            # Should be instantiable and usable
            import torch
            comp = cls()
            feat = torch.randn(10, 64)
            out = comp.compress(feat, 5)
            assert out.shape == (5, 64)
        finally:
            os.unlink(tmp_path)

    def test_load_external_file_not_found(self):
        """Loading from nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_external_compressor("/nonexistent/path.py")

    def test_load_external_no_compressor(self):
        """File without a Compressor subclass raises ValueError."""
        code = "x = 42\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir="."
        ) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="No Compressor subclass"):
                load_external_compressor(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestBackendDiscovery:

    def test_gemma4_supported(self):
        """Gemma 4 model IDs should resolve to the gemma4 backend."""
        cls = discover_backend("google/gemma-4-E4B-it")
        assert issubclass(cls, ModelBackend)
        assert cls.name == "gemma4"

    def test_gemma4_variants(self):
        """All Gemma 4 variant strings should match."""
        for model_id in [
            "google/gemma-4-E2B-it",
            "google/gemma-4-E4B-it",
            "google/gemma-4-E12B-it",
            "google/gemma-4-E27B-it",
            "google/gemma4-e4b",
        ]:
            cls = discover_backend(model_id)
            assert cls.name == "gemma4"

    def test_unknown_model_raises(self):
        """Unknown model ID should raise ValueError with helpful message."""
        with pytest.raises(ValueError, match="No backend found"):
            discover_backend("openai/gpt-5-vision")

    def test_model_catalog(self):
        """Gemma 4 backend should have a MODELS catalog."""
        cls = discover_backend("google/gemma-4-E4B-it")
        assert hasattr(cls, "MODELS")
        assert "gemma-4-E4B-it" in cls.MODELS
        info = cls.MODELS["gemma-4-E4B-it"]
        assert "hf_id" in info
        assert "quantizations" in info

    def test_resolve_short_name(self):
        """Short model names should resolve to full HF IDs."""
        cls = discover_backend("google/gemma-4-E4B-it")
        assert cls.resolve_model_id("gemma-4-E4B-it") == "google/gemma-4-E4B-it"


class TestDatasetDiscovery:

    def test_discovers_builtins(self):
        """Built-in datasets should be discovered."""
        from vtbench.datasets import discover_datasets
        datasets = discover_datasets()
        assert "mmmu_pro" in datasets
        assert "gqa" in datasets

    def test_entries_have_required_fields(self):
        """Every dataset entry must have name, description."""
        from vtbench.datasets import discover_datasets
        for name, entry in discover_datasets().items():
            assert entry.name
            assert entry.description
            assert hasattr(entry, "fetch")
            assert hasattr(entry, "load")

    def test_list_backends(self):
        """list_backends returns available backend folder names."""
        backends = list_backends()
        assert "gemma4" in backends
        assert isinstance(backends, list)
