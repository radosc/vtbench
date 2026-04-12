"""Dataset registry — fetchable benchmark datasets."""

from vtbench.datasets._base import DatasetEntry, load_manifest, DATA_HOME
from vtbench.datasets._discover import discover_datasets

__all__ = ["DatasetEntry", "load_manifest", "discover_datasets", "DATA_HOME"]
