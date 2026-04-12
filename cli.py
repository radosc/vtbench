"""Command-line interface for VTBench.

Usage:
    python -m vtbench list
    python -m vtbench fetch mmmu_pro
    python -m vtbench run --model gemma-4-E4B-it --image photo.jpg
    python -m vtbench benchmark --model gemma-4-E4B-it --data mmmu_pro ...
    python -m vtbench benchmark --config experiment.json

See --help for each subcommand.
"""

import argparse
import json
import sys


def cmd_list(args):
    """List available models, datasets, and compressors."""
    from vtbench.compressors._discover import discover_compressors
    from vtbench.models._discover import list_backends
    from vtbench.datasets import discover_datasets
    from importlib import import_module

    # Models
    print("\nModels:", flush=True)
    for backend_name in list_backends():
        try:
            module = import_module(f"vtbench.models.{backend_name}")
            cls = getattr(module, "Backend", None)
            if cls and hasattr(cls, "MODELS") and cls.MODELS:
                for short_name, info in cls.MODELS.items():
                    quants = ", ".join(info.get("quantizations", []))
                    print(
                        f"  {short_name:<22s} {info['description']:<40s} "
                        f"{info.get('vram_bf16', ''):>8s}  [{quants}]"
                    )
        except ImportError:
            print(f"  {backend_name} (could not load)")

    # Datasets
    print("\nDatasets:", flush=True)
    datasets = discover_datasets()
    if datasets:
        for name, entry in sorted(datasets.items()):
            status = "fetched" if entry.is_fetched else "not fetched"
            n = f"~{entry.n_samples_approx}" if entry.n_samples_approx else "?"
            print(f"  {name:<16s} {entry.description:<52s} [{status}, {n} samples]")
    else:
        print("  (none found)")

    # Compressors
    print("\nCompressors:", flush=True)
    compressors = discover_compressors()
    if compressors:
        for name, cls in sorted(compressors.items()):
            desc = getattr(cls, "description", "") or ""
            print(f"  {name:<16s} {desc}")
    else:
        print("  (none found)")

    print()


def cmd_fetch(args):
    """Fetch a dataset."""
    from vtbench.datasets import discover_datasets

    datasets = discover_datasets()
    if args.name not in datasets:
        print(f"Unknown dataset '{args.name}'.", file=sys.stderr)
        print(f"Available: {list(datasets.keys())}", file=sys.stderr)
        sys.exit(1)

    entry = datasets[args.name]

    kwargs = {}
    if args.n:
        kwargs["n_samples"] = args.n
    if args.seed:
        kwargs["seed"] = args.seed
    if args.source:
        kwargs["source"] = args.source

    path = entry.fetch(**kwargs)
    print(f"\nManifest: {path}")


def cmd_run(args):
    """Run inference on a single image."""
    from PIL import Image
    from vtbench.pipeline import Pipeline

    compressor = _resolve_compressor(args.compressor) if args.compressor else None
    model_id = _resolve_model_id(args.model)

    pipe = Pipeline(model_id)
    image = Image.open(args.image).convert("RGB")

    answer = pipe(
        image, args.prompt,
        compressor=compressor,
        ratio=args.ratio,
        gen_config={"max_new_tokens": args.max_tokens} if args.max_tokens else None,
    )

    n_native = pipe.native_token_count(image)
    n_used = int(n_native * args.ratio) if compressor else n_native
    label = compressor.name if compressor else "stock"

    print(f"\n[{label} | {n_used}/{n_native} tokens | ratio={args.ratio:.2f}]")
    print(answer)
    print()


def cmd_benchmark(args):
    """Run a benchmark sweep."""
    from vtbench.benchmark.runner import BenchmarkRunner, BenchmarkConfig

    # Load from config file if provided
    if args.config:
        cfg = json.loads(open(args.config).read())
        model_id = _resolve_model_id(cfg["model"])
        compressors = [_resolve_compressor(c) for c in cfg.get("compressors", ["divprune"])]
        ratios = cfg.get("ratios", [0.5, 0.25])
        gen_config = cfg.get("gen_config", None)

        config = BenchmarkConfig(
            model_id=model_id,
            data=cfg.get("data", cfg.get("dataset", "")),
            compressors=compressors,
            ratios=ratios,
            n_samples=cfg.get("n_samples", 0),
            seed=cfg.get("seed", 42),
            source=cfg.get("source", ""),
            output_dir=cfg.get("output_dir", "results/benchmark"),
            gen_config=gen_config,
        )
    else:
        # From CLI flags
        compressors = [_resolve_compressor(c) for c in args.compressors]
        ratios = [float(r) for r in args.ratios]

        config = BenchmarkConfig(
            model_id=_resolve_model_id(args.model),
            data=args.data,
            compressors=compressors,
            ratios=ratios,
            n_samples=args.n,
            seed=args.seed,
            source=args.source or "",
            output_dir=args.output,
        )

    runner = BenchmarkRunner()
    runner.run(config)


def _resolve_compressor(name_or_path: str):
    """Resolve a compressor by built-in name or external file path."""
    if name_or_path.endswith(".py") or "/" in name_or_path or "\\" in name_or_path:
        from vtbench.compressors._discover import load_external_compressor
        cls = load_external_compressor(name_or_path)
        return cls()

    from vtbench.compressors._discover import discover_compressors
    available = discover_compressors()
    if name_or_path not in available:
        print(f"Error: unknown compressor '{name_or_path}'", file=sys.stderr)
        print(f"Available: {list(available.keys())}", file=sys.stderr)
        sys.exit(1)
    return available[name_or_path]()


def _resolve_model_id(model_id: str) -> str:
    """Resolve short model names to full HuggingFace IDs."""
    from vtbench.models._discover import resolve_model_id
    return resolve_model_id(model_id)


def main():
    parser = argparse.ArgumentParser(
        prog="vtbench",
        description="VTBench — Visual Token Compression Benchmark for VLMs",
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- list ---
    subparsers.add_parser("list", help="List available models, datasets, and compressors")

    # --- fetch ---
    p_fetch = subparsers.add_parser("fetch", help="Download and prepare a dataset")
    p_fetch.add_argument("name", help="Dataset name (see `vtbench list`)")
    p_fetch.add_argument("--source", default=None,
                         help="Source directory (required for some datasets, e.g. gqa)")
    p_fetch.add_argument("--n", type=int, default=0,
                         help="Max samples to include (0 = all)")
    p_fetch.add_argument("--seed", type=int, default=42,
                         help="Random seed for sample selection")

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run inference on a single image")
    p_run.add_argument("--model", required=True, help="Model name or HuggingFace ID")
    p_run.add_argument("--image", required=True, help="Path to image file")
    p_run.add_argument("--prompt", default="Describe this image.",
                       help="Text prompt / question")
    p_run.add_argument("--compressor", default=None,
                       help="Compressor name or path to .py file (omit for stock)")
    p_run.add_argument("--ratio", type=float, default=0.5,
                       help="Token retention ratio (0.5 = 2x compression)")
    p_run.add_argument("--max-tokens", type=int, default=None,
                       help="Override max_new_tokens for generation")

    # --- benchmark ---
    p_bench = subparsers.add_parser("benchmark", help="Run a benchmark sweep")
    p_bench.add_argument("--config", default=None,
                         help="JSON config file (replaces all other flags)")
    p_bench.add_argument("--model", default=None, help="Model name or HuggingFace ID")
    p_bench.add_argument("--data", default=None,
                         help="Dataset name (from `vtbench list`) or path to manifest.jsonl")
    p_bench.add_argument("--source", default=None,
                         help="Source directory (for datasets like gqa)")
    p_bench.add_argument("--compressors", nargs="+", default=None,
                         help="Compressor names or .py file paths")
    p_bench.add_argument("--ratios", nargs="+", default=["0.5", "0.25"],
                         help="Token retention ratios (default: 0.5 0.25)")
    p_bench.add_argument("--n", type=int, default=0,
                         help="Number of samples, 0 = all (default: 0)")
    p_bench.add_argument("--seed", type=int, default=42,
                         help="Random seed (default: 42)")
    p_bench.add_argument("--output", default="results/benchmark",
                         help="Output directory (default: results/benchmark)")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "benchmark":
        if not args.config and (not args.model or not args.data or not args.compressors):
            print("Error: --config or (--model, --data, --compressors) required",
                  file=sys.stderr)
            p_bench.print_help()
            sys.exit(1)
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
