"""Auto-discovery for dataset entries."""

import importlib
from pathlib import Path
from vtbench.datasets._base import DatasetEntry


def discover_datasets() -> dict[str, DatasetEntry]:
    """Scan this directory for DatasetEntry subclasses. Returns instances."""
    found = {}
    datasets_dir = Path(__file__).parent

    for path in sorted(datasets_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"vtbench.datasets.{path.stem}")
        except ImportError:
            continue
        for obj in vars(module).values():
            if (isinstance(obj, type)
                    and issubclass(obj, DatasetEntry)
                    and obj is not DatasetEntry
                    and obj.name):
                found[obj.name] = obj()

    return found
