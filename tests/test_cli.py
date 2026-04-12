"""Tests for CLI argument parsing and compressor resolution.

No model loading — tests the argument parsing and resolution logic only.
"""

import pytest
import subprocess
import sys


class TestCLI:

    def test_list_command(self):
        """python -m vtbench list should succeed and show models, datasets, compressors."""
        result = subprocess.run(
            [sys.executable, "-m", "vtbench", "list"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        # Models
        assert "gemma-4-E4B-it" in result.stdout
        # Datasets
        assert "mmmu_pro" in result.stdout
        assert "gqa" in result.stdout
        # Compressors
        assert "divprune" in result.stdout
        assert "fps" in result.stdout
        assert "identity" in result.stdout

    def test_no_command_shows_help(self):
        """Running without a command should show help."""
        result = subprocess.run(
            [sys.executable, "-m", "vtbench"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "list" in result.stdout or "usage" in result.stdout.lower()

    def test_run_help(self):
        """python -m vtbench run --help should show options."""
        result = subprocess.run(
            [sys.executable, "-m", "vtbench", "run", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--model" in result.stdout
        assert "--compressor" in result.stdout

    def test_benchmark_help(self):
        """python -m vtbench benchmark --help should show options."""
        result = subprocess.run(
            [sys.executable, "-m", "vtbench", "benchmark", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--compressors" in result.stdout
        assert "--ratios" in result.stdout
        assert "--data" in result.stdout
        assert "--config" in result.stdout


    def test_fetch_help(self):
        """python -m vtbench fetch --help should show options."""
        result = subprocess.run(
            [sys.executable, "-m", "vtbench", "fetch", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--source" in result.stdout


class TestModelResolution:

    def test_short_name_resolves(self):
        from vtbench.cli import _resolve_model_id
        assert _resolve_model_id("gemma-4-E4B-it") == "google/gemma-4-E4B-it"

    def test_full_id_passthrough(self):
        from vtbench.cli import _resolve_model_id
        assert _resolve_model_id("google/gemma-4-E4B-it") == "google/gemma-4-E4B-it"

    def test_unknown_passthrough(self):
        from vtbench.cli import _resolve_model_id
        assert _resolve_model_id("some/unknown-model") == "some/unknown-model"


class TestCompressorResolution:

    def test_resolve_builtin(self):
        from vtbench.cli import _resolve_compressor
        comp = _resolve_compressor("divprune")
        assert comp.name == "divprune"

    def test_resolve_builtin_fps(self):
        from vtbench.cli import _resolve_compressor
        comp = _resolve_compressor("fps")
        assert comp.name == "fps"

    def test_resolve_unknown_exits(self):
        """Unknown compressor name should exit with error."""
        from vtbench.cli import _resolve_compressor
        with pytest.raises(SystemExit):
            _resolve_compressor("nonexistent_algo")
