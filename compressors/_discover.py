"""Auto-discovery for compressor algorithms.

Scans vtbench/compressors/ for .py files containing Compressor subclasses.
Files prefixed with underscore (_base.py, _template.py, etc.) are skipped.
"""

import importlib
import importlib.util
from pathlib import Path
from vtbench.compressors._base import Compressor


def discover_compressors() -> dict[str, type[Compressor]]:
    """Scan this directory for Compressor subclasses.

    Returns dict mapping compressor.name -> class.
    """
    found = {}
    compressors_dir = Path(__file__).parent

    for path in sorted(compressors_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"vtbench.compressors.{path.stem}")
        except ImportError:
            continue
        for obj in vars(module).values():
            if (isinstance(obj, type)
                    and issubclass(obj, Compressor)
                    and obj is not Compressor
                    and obj.name):
                found[obj.name] = obj

    return found


def load_external_compressor(file_path: str) -> type[Compressor]:
    """Load a Compressor subclass from an external .py file.

    Usage: point at any .py file containing a Compressor subclass.
    The file doesn't need to be inside the vtbench package.

    Raises ValueError if no Compressor subclass is found in the file.
    """
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Compressor file not found: {path}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for obj in vars(module).values():
        if (isinstance(obj, type)
                and issubclass(obj, Compressor)
                and obj is not Compressor
                and obj.name):
            return obj

    raise ValueError(
        f"No Compressor subclass found in {path}.\n"
        f"Your file must contain a class that inherits from Compressor "
        f"and sets a `name` attribute."
    )


def list_compressors() -> list[str]:
    """Return names of all discovered built-in compressors."""
    return sorted(discover_compressors().keys())
