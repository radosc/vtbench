"""MMMU-Pro dataset — auto-fetching from HuggingFace Hub.

`vtbench fetch mmmu_pro` downloads the dataset, filters to single-image
samples, saves images to disk, and produces a JSONL manifest.

Requires: pip install datasets  (or: pip install "vtbench[mmmu]")
"""

import json
import re
import ast
import numpy as np
from pathlib import Path
from PIL import Image

from vtbench.datasets._base import DatasetEntry


class MMMUPro(DatasetEntry):

    name = "mmmu_pro"
    description = "MMMU-Pro multiple choice — academic/technical visual reasoning"
    n_samples_approx = 1600

    def fetch(self, n_samples: int = 0, seed: int = 42) -> Path:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "MMMU-Pro requires the `datasets` library.\n"
                "Install: pip install datasets\n"
                "    or:  pip install \"vtbench[mmmu]\""
            )

        print(f"Fetching MMMU-Pro from HuggingFace Hub...", flush=True)
        ds = load_dataset("MMMU/MMMU_Pro", "standard (4 options)", split="test")
        print(f"  Downloaded {len(ds)} samples", flush=True)

        # Prepare output directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        images_dir = self.data_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Filter to single-image samples and save
        all_entries = []
        for i, row in enumerate(ds):
            if row.get("image_1") is None or row.get("image_2") is not None:
                continue
            img = row["image_1"]
            if not isinstance(img, Image.Image):
                continue

            # Parse options
            options = row["options"]
            if isinstance(options, str):
                try:
                    options = json.loads(options)
                except json.JSONDecodeError:
                    options = ast.literal_eval(options)

            # Build prompt
            q_text = re.sub(r'<image\s*\d+>', '[image]', row["question"])
            opts = "\n".join(
                f"{chr(65 + j)}. {opt}" for j, opt in enumerate(options)
            )
            prompt = f"{q_text}\n\n{opts}\n\nRespond with ONLY the letter of the correct answer."

            # Save image
            img_name = f"{i:05d}.jpg"
            img.convert("RGB").save(images_dir / img_name, quality=95)

            all_entries.append({
                "id": f"mmmu_{i:05d}",
                "image": f"images/{img_name}",
                "prompt": prompt,
                "answer": row["answer"],
                "category": row.get("subject") or row.get("subfield") or "unknown",
            })

        # Subsample if requested
        if n_samples and n_samples < len(all_entries):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(all_entries), size=n_samples, replace=False)
            all_entries = [all_entries[i] for i in sorted(indices)]

        # Write manifest
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            for entry in all_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"  Saved {len(all_entries)} single-image samples to {self.data_dir}", flush=True)
        return self.manifest_path
