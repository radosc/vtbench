"""Pipeline — the single entry point connecting models and compressors.

The pipeline is a thin orchestrator. It owns no model-specific logic
and no algorithm-specific logic. Its job:
  1. Discover and load the right model backend for a given model_id
  2. Route calls through extract → compress → inject-generate

Usage:
    from vtbench import Pipeline
    from vtbench.compressors.divprune import DivPrune

    pipe = Pipeline("google/gemma-4-E4B-it")

    # Stock inference (no compression)
    answer = pipe(image, "What is in this image?")

    # Compressed inference
    answer = pipe(image, "What is in this image?",
                  compressor=DivPrune(), ratio=0.5)
"""

from typing import Optional
from PIL import Image
from torch import Tensor

from vtbench.compressors._base import Compressor
from vtbench.models._base import ModelBackend
from vtbench.models._discover import discover_backend


class Pipeline:
    """Connects a model backend with token compressors."""

    def __init__(self, model_id: str, **model_kwargs):
        """Initialize pipeline: discover and load the appropriate backend.

        Args:
            model_id:      HuggingFace model identifier
                           (e.g. "google/gemma-4-E4B-it")
            **model_kwargs: passed to backend.load() — dtype, device_map, etc.
        """
        backend_cls = discover_backend(model_id)
        self.backend: ModelBackend = backend_cls()
        self.backend.load(model_id, **model_kwargs)
        self.model_id = model_id

    def __call__(
        self,
        image: Image.Image,
        prompt: str,
        compressor: Optional[Compressor] = None,
        ratio: float = 1.0,
        gen_config: Optional[dict] = None,
    ) -> str:
        """Run inference — stock or compressed.

        Args:
            image:      PIL Image
            prompt:     text prompt / question
            compressor: a Compressor instance, or None for stock inference
            ratio:      fraction of tokens to keep (0.5 = 2x compression).
                        Ignored if compressor is None.
            gen_config: override generation parameters (max_new_tokens, etc.)

        Returns:
            Model's text answer.
        """
        if compressor is None or ratio >= 1.0:
            return self.backend.generate_stock(image, prompt, gen_config)

        if not 0.0 < ratio < 1.0:
            raise ValueError(
                f"ratio must be between 0 and 1 (exclusive), got {ratio}. "
                f"Use ratio=0.5 for 2x compression, ratio=0.25 for 4x."
            )

        # Extract native vision tokens
        features = self.backend.extract(image)
        n_native = len(features)

        # Compute target token count from ratio
        n_target = max(int(n_native * ratio), 1)

        # Run compression (promote to float for stable math)
        compressed = compressor.compress(features.float(), n_target)

        return self.backend.generate_compressed(
            image, prompt, compressed, gen_config
        )

    def extract(self, image: Image.Image) -> Tensor:
        """Extract raw vision tokens (for inspection / debugging)."""
        return self.backend.extract(image)

    def native_token_count(self, image: Image.Image) -> int:
        """How many vision tokens the model produces for this image."""
        return self.backend.native_token_count(image)

    def __repr__(self) -> str:
        return f"Pipeline(model={self.model_id!r}, backend={self.backend.name!r})"
