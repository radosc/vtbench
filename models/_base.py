"""Model backend base class — the blueprint for VLM integrations.

To add support for a new model family (e.g. Gemma 5, LLaVA, Qwen-VL):
  1. Create a subfolder: vtbench/models/your_model/
  2. Implement a class that inherits ModelBackend in backend.py
  3. Export it as `Backend` in __init__.py
  4. Done. The framework discovers it via the supports() classmethod.

All model-specific details (tokenization format, vision encoder quirks,
output parsing, monkey-patch mechanics) stay inside the backend.
Compressors and the benchmark runner never see them.
"""

from abc import ABC, abstractmethod
from typing import Optional
from torch import Tensor
from PIL import Image


class ModelBackend(ABC):
    """Abstract base for vision-language model integrations."""

    # Human-readable name for logging and result tables.
    name: str = ""

    # Model catalog: dict of short_name -> {hf_id, description, vram_bf16, quantizations}.
    # Populated by each backend. Shown by `vtbench list`.
    MODELS: dict = {}

    @classmethod
    @abstractmethod
    def supports(cls, model_id: str) -> bool:
        """Return True if this backend can handle the given HuggingFace model ID.

        Called during discovery. Should be a cheap string check, not a model load.
        Example: return "gemma-4" in model_id.lower()
        """
        ...

    @abstractmethod
    def load(self, model_id: str, **kwargs) -> None:
        """Load model, tokenizer, and processor into memory.

        Called once at pipeline construction. Store everything on self.
        kwargs may include dtype, device_map, quantization settings, etc.
        """
        ...

    @abstractmethod
    def extract(self, image: Image.Image) -> Tensor:
        """Extract vision tokens from an image.

        Returns:
            [N, D] float tensor — the raw vision token embeddings
            before any compression. N varies by image resolution and
            model config. D is the model's vision hidden dimension.
        """
        ...

    @abstractmethod
    def native_token_count(self, image: Image.Image) -> int:
        """Count how many vision tokens this model produces for this image.

        Used to compute compression targets from ratios.
        May be cheaper than extract() if the count can be derived
        from the preprocessor output alone.
        """
        ...

    @abstractmethod
    def generate_stock(self, image: Image.Image, prompt: str,
                       gen_config: Optional[dict] = None) -> str:
        """Run unmodified inference. Returns the model's parsed text answer."""
        ...

    @abstractmethod
    def generate_compressed(self, image: Image.Image, prompt: str,
                            features: Tensor,
                            gen_config: Optional[dict] = None) -> str:
        """Inject pre-compressed vision features and generate.

        The backend is responsible for:
          - Building input_ids with the correct number of image placeholders
          - Intercepting the vision encoder to return `features` instead
          - Restoring the original vision encoder after generation
          - Parsing the output into clean text

        Args:
            features: [M, D] compressed vision tokens (from a Compressor)
            gen_config: overrides for generation parameters

        Returns:
            Parsed text answer.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
