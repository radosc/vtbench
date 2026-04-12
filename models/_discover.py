"""Auto-discovery for model backends.

Scans vtbench/models/ for subfolders containing a Backend class.
Each subfolder is a self-contained model integration.
"""

from importlib import import_module
from pathlib import Path
from vtbench.models._base import ModelBackend


def discover_backend(model_id: str) -> type[ModelBackend]:
    """Find a backend that supports the given model ID.

    Scans each subfolder in vtbench/models/ for an __init__.py that
    exports a `Backend` class with a supports() classmethod.

    Raises ValueError with available backends if none matches.
    """
    models_dir = Path(__file__).parent
    available = []

    for subdir in sorted(models_dir.iterdir()):
        if not subdir.is_dir() or subdir.name.startswith("_"):
            continue
        available.append(subdir.name)
        try:
            module = import_module(f"vtbench.models.{subdir.name}")
        except ImportError:
            continue
        cls = getattr(module, "Backend", None)
        if cls is not None and issubclass(cls, ModelBackend) and cls.supports(model_id):
            return cls

    raise ValueError(
        f"No backend found for model '{model_id}'.\n"
        f"Available backends: {available}\n"
        f"To add support, create vtbench/models/<name>/ with a Backend class."
    )


def list_backends() -> list[str]:
    """Return names of all available model backend folders."""
    models_dir = Path(__file__).parent
    return [
        d.name for d in sorted(models_dir.iterdir())
        if d.is_dir() and not d.name.startswith("_")
    ]


def resolve_model_id(model_id: str) -> str:
    """Resolve a short model name to a full HuggingFace ID.

    Scans all backend MODELS catalogs for a matching short name.
    Returns the input unchanged if no match found (assumed to be a full ID).
    """
    for name in list_backends():
        try:
            module = import_module(f"vtbench.models.{name}")
        except ImportError:
            continue
        cls = getattr(module, "Backend", None)
        if cls and hasattr(cls, "MODELS") and model_id in cls.MODELS:
            return cls.MODELS[model_id]["hf_id"]
    return model_id
