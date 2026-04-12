"""Dataset entry base class.

To add a new dataset:
  1. Create a .py file in this folder (e.g. my_dataset.py)
  2. Subclass DatasetEntry, implement fetch() and load()
  3. Done. `vtbench list` shows it, `vtbench fetch <name>` downloads it.
"""

from abc import ABC, abstractmethod
from pathlib import Path

# All fetched datasets live here
DATA_HOME = Path.home() / ".vtbench" / "datasets"


class DatasetEntry(ABC):
    """Registry entry for a fetchable benchmark dataset."""

    # Identifier used in CLI and config files.
    name: str = ""

    # One-line description for `vtbench list`.
    description: str = ""

    # Number of samples after filtering (approximate, for display).
    n_samples_approx: int = 0

    @property
    def data_dir(self) -> Path:
        """Local directory where this dataset is stored after fetch."""
        return DATA_HOME / self.name

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.jsonl"

    @property
    def is_fetched(self) -> bool:
        return self.manifest_path.exists()

    @abstractmethod
    def fetch(self, n_samples: int = 0, seed: int = 42) -> Path:
        """Download and prepare the dataset.

        Produces a JSONL manifest at self.manifest_path where each line is:
            {"id": "...", "image": "relative/path.jpg", "prompt": "...",
             "answer": "...", "category": "..."}

        Image paths are relative to the manifest's parent directory.

        Args:
            n_samples: max samples to include (0 = all available).
            seed: random seed for deterministic sample selection.

        Returns:
            Path to the manifest file.
        """
        ...

    def load(self, n_samples: int = 0, seed: int = 42) -> list[dict]:
        """Load samples from the manifest. Fetches if not already done."""
        if not self.is_fetched:
            self.fetch(n_samples=n_samples, seed=seed)
        return load_manifest(self.manifest_path, n_samples=n_samples, seed=seed)

    def __repr__(self) -> str:
        status = "fetched" if self.is_fetched else "not fetched"
        return f"{self.name} [{status}]"


def load_manifest(manifest_path: Path, n_samples: int = 0,
                  seed: int = 42) -> list[dict]:
    """Load samples from a JSONL manifest file.

    Each line must have: id, image (relative path), prompt, answer.
    Optional: category.

    Returns list of dicts with PIL Image objects loaded.
    """
    import json
    import numpy as np
    from PIL import Image

    manifest_path = Path(manifest_path)
    base_dir = manifest_path.parent

    all_samples = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            img_path = base_dir / entry["image"]
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            all_samples.append({
                "id": entry.get("id", str(len(all_samples))),
                "image": img,
                "prompt": entry["prompt"],
                "answer": entry.get("answer", ""),
                "category": entry.get("category", ""),
            })

    if n_samples and n_samples < len(all_samples):
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(all_samples), size=n_samples, replace=False)
        all_samples = [all_samples[i] for i in sorted(indices)]

    return all_samples
