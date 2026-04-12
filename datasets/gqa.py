"""GQA dataset — visual question answering.

GQA images must be downloaded manually (hosted on Google Cloud):
  1. Download from https://cs.stanford.edu/people/dorarad/gqa/download.html
  2. Extract to a directory with this layout:
       gqa_dir/
         images/           *.jpg files
         val_balanced_questions.json   (or testdev_balanced_questions.json)
  3. Run: vtbench fetch gqa --source /path/to/gqa_dir

This creates a JSONL manifest that VTBench can use for benchmarking.
"""

import json
import os
import shutil
import numpy as np
from pathlib import Path

from vtbench.datasets._base import DatasetEntry


class GQA(DatasetEntry):

    name = "gqa"
    description = "GQA visual QA — open-ended scene understanding"
    n_samples_approx = 132000

    def fetch(self, n_samples: int = 0, seed: int = 42, source: str = "") -> Path:
        if not source:
            raise ValueError(
                "GQA requires a local source directory.\n"
                "Download from: https://cs.stanford.edu/people/dorarad/gqa/download.html\n"
                "Then run: vtbench fetch gqa --source /path/to/gqa_dir\n\n"
                "Expected layout:\n"
                "  gqa_dir/\n"
                "    images/              (*.jpg files)\n"
                "    val_balanced_questions.json"
            )

        source_dir = Path(source)
        questions_path = source_dir / "val_balanced_questions.json"
        if not questions_path.exists():
            questions_path = source_dir / "testdev_balanced_questions.json"
        if not questions_path.exists():
            raise FileNotFoundError(
                f"No questions file in {source_dir}.\n"
                f"Expected val_balanced_questions.json or testdev_balanced_questions.json"
            )

        images_dir = source_dir / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"No images/ directory in {source_dir}")

        print(f"Loading GQA questions from {questions_path}...", flush=True)
        raw = json.loads(questions_path.read_text())

        # Prepare output — symlink images instead of copying
        self.data_dir.mkdir(parents=True, exist_ok=True)
        out_images = self.data_dir / "images"
        if not out_images.exists():
            # Try symlink, fall back to noting the source path
            try:
                out_images.symlink_to(images_dir.resolve())
                print(f"  Linked images from {images_dir}", flush=True)
            except OSError:
                # Windows without developer mode can't symlink
                # Just store the absolute path and use it directly
                out_images.mkdir(exist_ok=True)
                (self.data_dir / ".images_source").write_text(str(images_dir.resolve()))
                print(f"  Images source: {images_dir} (will read from there)", flush=True)

        # Determine where images actually are
        if out_images.is_symlink() or any(out_images.iterdir()):
            actual_images_dir = out_images
        elif (self.data_dir / ".images_source").exists():
            actual_images_dir = Path((self.data_dir / ".images_source").read_text().strip())
        else:
            actual_images_dir = images_dir

        # Categorize and build entries
        all_entries = []
        for qid, q in raw.items():
            img_path = actual_images_dir / f"{q['imageId']}.jpg"
            if not img_path.exists():
                continue

            qt = q["question"].lower()
            if qt.startswith("how many"):
                category = "count"
            elif qt.startswith("what color"):
                category = "color"
            elif qt.startswith("is there") or qt.startswith("are there"):
                category = "exist"
            elif qt.startswith("what is") or qt.startswith("what are"):
                category = "what"
            elif qt.startswith("where"):
                category = "where"
            elif " or " in qt.split("?")[0]:
                category = "choice"
            else:
                category = "other"

            # Use relative path if symlinked, absolute if not
            if out_images.is_symlink() or (out_images.exists() and any(out_images.iterdir())):
                img_ref = f"images/{q['imageId']}.jpg"
            else:
                img_ref = str(img_path)

            all_entries.append({
                "id": qid,
                "image": img_ref,
                "prompt": q["question"] + " Answer with a single word or short phrase.",
                "answer": q.get("answer", "").lower().strip(),
                "category": category,
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

        print(f"  Saved {len(all_entries)} samples to {self.data_dir}", flush=True)
        return self.manifest_path
