"""Benchmark runner — iterate datasets x compressors x ratios.

Produces structured results with:
  - Per-sample answers for every configuration
  - Accuracy tables with stock baseline comparison
  - Per-category breakdown
  - Token counts (native vs compressed)
  - JSON checkpoint for resumption

Usage:
    from vtbench.benchmark.runner import BenchmarkRunner, BenchmarkConfig
    from vtbench.compressors.divprune import DivPrune
    from vtbench.compressors.fps import FPS

    config = BenchmarkConfig(
        model_id="google/gemma-4-E4B-it",
        data="mmmu_pro",
        compressors=[DivPrune(), FPS()],
        ratios=[0.5, 0.25],
    )
    runner = BenchmarkRunner()
    results = runner.run(config)
"""

import json
import time
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from collections import defaultdict

from vtbench.pipeline import Pipeline
from vtbench.compressors._base import Compressor
from vtbench.benchmark.scoring import evaluate_answer


@dataclass
class BenchmarkConfig:
    model_id: str
    data: str                                 # dataset name (from registry) or path to manifest.jsonl
    compressors: list[Compressor]
    ratios: list[float] = field(default_factory=lambda: [0.5, 0.25])
    n_samples: int = 0                        # 0 = use all available samples
    seed: int = 42
    source: str = ""                          # source dir for datasets that need it (e.g. gqa)
    output_dir: str = "results/benchmark"
    gen_config: Optional[dict] = None


class BenchmarkRunner:

    def run(self, config: BenchmarkConfig) -> dict:
        """Execute the full benchmark sweep.

        Flow:
          1. Load dataset (from registry name or manifest path)
          2. Load model (once, auto-downloads from HuggingFace)
          3. For each sample: run stock + all (compressor, ratio) pairs
          4. Aggregate and print results table with per-category breakdown
          5. Save JSON results + checkpoint

        Returns:
            Summary dict with accuracy tables.
        """
        out_dir = Path(config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        samples = self._load_data(config)
        print(f"  {len(samples)} samples loaded", flush=True)

        # Load model
        model_id = self._resolve_model(config.model_id)
        print(f"Loading model: {model_id}", flush=True)
        pipe = Pipeline(model_id)
        print(f"  Backend: {pipe.backend.name}", flush=True)

        # Check for existing checkpoint
        ckpt_path = out_dir / "checkpoint.json"
        done_ids = set()
        all_results = []
        if ckpt_path.exists():
            all_results = json.loads(ckpt_path.read_text())
            done_ids = {r["id"] for r in all_results}
            print(f"  Resuming: {len(done_ids)} samples already done", flush=True)

        # Build configuration labels
        configs = [("stock", None, 1.0)]
        for compressor in config.compressors:
            for ratio in config.ratios:
                label = f"{compressor.name}_{ratio:.2f}"
                configs.append((label, compressor, ratio))

        # Run
        t_start = time.time()
        n_new = 0

        for i, sample in enumerate(samples):
            if sample["id"] in done_ids:
                continue

            result = {"id": sample["id"], "answer_gt": sample["answer"],
                      "category": sample.get("category", "")}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Get native token count once per image
            n_native = pipe.native_token_count(sample["image"])
            result["n_native"] = n_native

            for label, compressor, ratio in configs:
                n_tokens = n_native if compressor is None else max(int(n_native * ratio), 1)
                try:
                    answer = pipe(
                        sample["image"], sample["prompt"],
                        compressor=compressor, ratio=ratio,
                        gen_config=config.gen_config,
                    )
                except Exception as e:
                    answer = ""
                    result.setdefault("errors", []).append({"config": label, "error": str(e)})

                is_correct = evaluate_answer(answer, sample["answer"]) if sample["answer"] and answer else None
                result[label] = {
                    "answer": answer[:200],
                    "correct": is_correct,
                    "n_tokens": n_tokens,
                }

            all_results.append(result)
            n_new += 1
            n_total = len(all_results)

            # Progress
            stock_ans = result["stock"]["answer"][:60]
            elapsed_total = time.time() - t_start
            rate = n_new / elapsed_total if elapsed_total > 0 else 0
            remaining = (len(samples) - n_total) / rate if rate > 0 else 0
            print(
                f"  [{n_total:4d}/{len(samples)}] "
                f"id={sample['id'][:12]:12s}  "
                f"stock={stock_ans:60s}  "
                f"[{elapsed_total:.0f}s elapsed, ~{remaining:.0f}s remaining]",
                flush=True,
            )

            # Checkpoint every 10 samples
            if n_new % 10 == 0 or n_total == len(samples):
                ckpt_path.write_text(
                    json.dumps(all_results, indent=2, ensure_ascii=False)
                )

        # Aggregate results
        summary = self._aggregate(all_results, configs, config)

        # Print table
        self._print_table(summary, config)

        # Save
        results_path = out_dir / "results.json"
        results_path.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False)
        )
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))

        print(f"\nResults saved to {out_dir}/", flush=True)
        return summary

    def _load_data(self, config: BenchmarkConfig) -> list[dict]:
        """Load from dataset registry name or manifest file path."""
        data = config.data

        # Check if it's a file path
        if Path(data).suffix in (".jsonl", ".json") and Path(data).exists():
            print(f"Loading manifest: {data}", flush=True)
            from vtbench.datasets._base import load_manifest
            return load_manifest(Path(data), n_samples=config.n_samples, seed=config.seed)

        # Check dataset registry
        from vtbench.datasets import discover_datasets
        datasets = discover_datasets()
        if data in datasets:
            entry = datasets[data]
            print(f"Loading dataset: {entry.name} — {entry.description}", flush=True)
            if not entry.is_fetched:
                print(f"  Dataset not fetched yet, downloading...", flush=True)
                entry.fetch(n_samples=config.n_samples, seed=config.seed,
                            **{"source": config.source} if config.source else {})
            return entry.load(n_samples=config.n_samples, seed=config.seed)

        # Check if it's a directory of images (qualitative mode)
        if Path(data).is_dir():
            print(f"Loading image folder: {data}", flush=True)
            from PIL import Image
            extensions = {".jpg", ".jpeg", ".png", ".webp"}
            samples = []
            for p in sorted(Path(data).iterdir()):
                if p.suffix.lower() not in extensions:
                    continue
                try:
                    img = Image.open(p).convert("RGB")
                except Exception:
                    continue
                samples.append({
                    "id": p.stem, "image": img,
                    "prompt": "Describe this image.",
                    "answer": "", "category": "custom",
                })
            return samples

        raise ValueError(
            f"Unknown dataset '{data}'.\n"
            f"Available datasets: {list(datasets.keys())}\n"
            f"Or provide a path to a .jsonl manifest file."
        )

    @staticmethod
    def _resolve_model(model_id: str) -> str:
        """Resolve short model names to full HuggingFace IDs."""
        from vtbench.models._discover import resolve_model_id
        return resolve_model_id(model_id)

    def _aggregate(self, results, configs, config):
        summary = {
            "model": config.model_id,
            "data": config.data,
            "n_samples": len(results),
            "seed": config.seed,
            "configs": {},
        }

        for label, _, ratio in configs:
            entries = [r[label] for r in results if label in r]
            scored = [e for e in entries if e["correct"] is not None]
            correct = sum(1 for e in scored if e["correct"])
            n_scored = len(scored)
            avg_tokens = (sum(e.get("n_tokens", 0) for e in entries) / len(entries)) if entries else 0

            summary["configs"][label] = {
                "ratio": ratio,
                "n_scored": n_scored,
                "correct": correct,
                "accuracy": round(correct / max(n_scored, 1) * 100, 1),
                "avg_tokens": round(avg_tokens),
            }

        # Per-category breakdown
        categories = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
        for r in results:
            cat = r.get("category", "other")
            for label, _, _ in configs:
                if label in r and r[label]["correct"] is not None:
                    categories[cat][label]["total"] += 1
                    if r[label]["correct"]:
                        categories[cat][label]["correct"] += 1

        summary["categories"] = {
            cat: {
                label: {
                    "correct": stats["correct"],
                    "total": stats["total"],
                    "accuracy": round(stats["correct"] / max(stats["total"], 1) * 100, 1),
                }
                for label, stats in labels.items()
            }
            for cat, labels in categories.items()
        }

        return summary

    def _print_table(self, summary, config):
        stock = summary["configs"].get("stock", {})
        stock_acc = stock.get("accuracy", 0)
        stock_tokens = stock.get("avg_tokens", 0)

        print(flush=True)
        print(f"{'=' * 72}", flush=True)
        print(
            f"  Model: {config.model_id}  |  "
            f"Data: {config.data} ({summary['n_samples']} samples)  |  "
            f"Seed: {config.seed}",
            flush=True,
        )
        print(f"{'=' * 72}", flush=True)
        header = (f"  {'Compressor':<20s} {'Ratio':>6s} {'Tokens':>10s} "
                  f"{'Accuracy':>10s} {'vs Stock':>10s}")
        print(header, flush=True)
        print(f"  {'-' * 66}", flush=True)

        for label, info in summary["configs"].items():
            acc = info["accuracy"]
            gap = f"{acc - stock_acc:+.1f}pp" if label != "stock" else "--"
            tokens = info.get("avg_tokens", 0)
            tok_str = f"{tokens:>4.0f}" if label == "stock" else f"{stock_tokens:.0f}->{tokens:.0f}"
            print(
                f"  {label:<20s} {info['ratio']:>6.2f} {tok_str:>10s} "
                f"{acc:>9.1f}% {gap:>10s}",
                flush=True,
            )

        print(f"{'=' * 72}", flush=True)

        # Per-category breakdown (only categories with >= 10 samples)
        categories = summary.get("categories", {})
        config_labels = list(summary["configs"].keys())
        printable_cats = {
            cat: data for cat, data in categories.items()
            if any(data.get(l, {}).get("total", 0) >= 10 for l in config_labels)
        }
        if printable_cats:
            print(flush=True)
            print(f"  Per-category breakdown (categories with >= 10 samples):", flush=True)
            cat_header = f"  {'Category':<16s}"
            for label in config_labels:
                cat_header += f" {label:>14s}"
            print(cat_header, flush=True)
            print(f"  {'-' * (16 + 15 * len(config_labels))}", flush=True)

            for cat in sorted(printable_cats.keys()):
                data = printable_cats[cat]
                row = f"  {cat:<16s}"
                for label in config_labels:
                    stats = data.get(label, {})
                    total = stats.get("total", 0)
                    if total >= 10:
                        acc = stats.get("accuracy", 0)
                        row += f" {acc:>5.1f}% ({total:>3d})"
                    else:
                        row += f" {'--':>14s}"
                print(row, flush=True)
            print(flush=True)
