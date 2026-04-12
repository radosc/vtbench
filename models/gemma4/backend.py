"""Gemma 4 model backend.

Handles all Gemma 4-specific mechanics:
  - Model loading with correct dtype/quantization constraints
  - Vision token extraction via SigLIP encoder
  - Monkey-patch injection of compressed tokens
  - Gemma 4 chat format output parsing

Every quirk and setting below was validated through hundreds of
benchmark runs. Comments explain WHY each setting exists.
"""

import torch
from typing import Optional
from PIL import Image

from vtbench.models._base import ModelBackend
from vtbench.models.gemma4.parsing import parse_gemma4_output


# ---------------------------------------------------------------------------
# Generation defaults — each one earned through painful debugging
# ---------------------------------------------------------------------------
GENERATION_DEFAULTS = {
    # Greedy decoding for reproducible benchmark results.
    # Sampling (do_sample=True) gives different results across runs,
    # making compression algorithm comparison unreliable.
    "do_sample": False,

    # CRITICAL: must be >= 1. Without this, Gemma 4 greedy decoding emits
    # EOS immediately on ~33% of images, producing empty answers. This was
    # the single most confusing bug in early SAVT development — the model
    # appeared to "refuse" answering, but it was just a greedy-decoding
    # degenerate mode. Setting min_new_tokens=1 forces at least one real
    # token before EOS is allowed.
    "min_new_tokens": 1,

    # Generous upper bound. Most VQA answers are <50 tokens.
    # Multiple-choice benchmarks need ~10. Open-ended needs more.
    # This is a safety ceiling, not a target.
    "max_new_tokens": 512,
}


class _FakeVisionOutput:
    """Minimal wrapper to match the return type of get_image_features().

    Gemma 4's forward pass expects vision features wrapped in an object
    with a .pooler_output attribute. This is the simplest conforming type.
    """
    def __init__(self, features):
        self.pooler_output = features


class Gemma4Backend(ModelBackend):
    """Model backend for the Google Gemma 4 family.

    Supports all Gemma 4 variants: E2B, E4B, E12B, E27B.
    Vision encoder: SigLIP (shared across all Gemma 4 sizes).
    """

    name = "gemma4"

    # Model catalog — shown by `vtbench list`, used for short-name resolution.
    # Short names (left) resolve to HuggingFace model IDs (hf_id).
    MODELS = {
        "gemma-4-E2B-it": {
            "hf_id": "google/gemma-4-E2B-it",
            "description": "Gemma 4 E2B (2B params, smallest)",
            "vram_bf16": "~6 GB",
            "quantizations": ["bf16", "8bit"],
        },
        "gemma-4-E4B-it": {
            "hf_id": "google/gemma-4-E4B-it",
            "description": "Gemma 4 E4B (4B params, recommended)",
            "vram_bf16": "~18 GB",
            "quantizations": ["bf16", "8bit"],
        },
        "gemma-4-E12B-it": {
            "hf_id": "google/gemma-4-E12B-it",
            "description": "Gemma 4 E12B (12B params)",
            "vram_bf16": "~42 GB",
            "quantizations": ["bf16", "8bit"],
        },
        "gemma-4-E27B-it": {
            "hf_id": "google/gemma-4-E27B-it",
            "description": "Gemma 4 E27B (27B params, largest)",
            "vram_bf16": "~65 GB",
            "quantizations": ["bf16", "8bit"],
        },
    }

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = None

    @classmethod
    def supports(cls, model_id: str) -> bool:
        lower = model_id.lower()
        return "gemma-4" in lower or "gemma4" in lower

    @classmethod
    def resolve_model_id(cls, short_name: str) -> str:
        """Resolve a short name to a full HuggingFace model ID.

        Accepts both short names ('gemma-4-E4B-it') and full IDs
        ('google/gemma-4-E4B-it'). Returns the full ID.
        """
        if short_name in cls.MODELS:
            return cls.MODELS[short_name]["hf_id"]
        # Already a full ID or unknown — pass through
        return short_name

    def load(self, model_id: str, **kwargs) -> None:
        """Load a Gemma 4 model.

        Keyword args:
            dtype:      torch dtype. Default: bfloat16 (Gemma 4's native
                        training precision). float16 also works but offers
                        no advantage. NEVER use float32 — wastes VRAM for
                        identical results.
            device_map: HuggingFace device placement. Default: "auto".
            load_in_8bit: Enable 8-bit quantization. Default: False.
                          8-bit is the MINIMUM viable quantization for
                          Gemma 4 vision. 4-bit quantization causes severe
                          hallucination on ALL vision tasks — the model
                          confidently describes objects, text, and scenes
                          that don't exist in the image. This was verified
                          across hundreds of test images. Do NOT use 4-bit.
            load_in_4bit: Enable 4-bit quantization. Default: False.
                          WARNING: 4-bit causes severe hallucination on
                          vision tasks. Included for completeness and for
                          users who want to verify this themselves.
        """
        # Lazy import — only depends on transformers at runtime
        from transformers import (
            AutoTokenizer,
            AutoProcessor,
            Gemma4ForConditionalGeneration,
        )

        dtype = kwargs.get("dtype", torch.bfloat16)
        device_map = kwargs.get("device_map", "auto")
        load_in_8bit = kwargs.get("load_in_8bit", False)
        load_in_4bit = kwargs.get("load_in_4bit", False)

        if load_in_4bit:
            import warnings
            warnings.warn(
                "4-bit quantization causes severe hallucination on Gemma 4 "
                "vision tasks. Results will be unreliable. Use 8-bit minimum "
                "for meaningful benchmarks.",
                UserWarning,
                stacklevel=2,
            )

        load_kwargs = {"device_map": device_map}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif load_in_8bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["torch_dtype"] = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            model_id, **load_kwargs
        )
        self.device = next(self.model.parameters()).device

    def extract(self, image: Image.Image) -> torch.Tensor:
        """Extract vision tokens from an image.

        Uses a minimal dummy prompt to trigger the processor's image
        preprocessing (resizing, normalization, tiling for SigLIP).
        The text content is irrelevant — we only use the vision outputs.

        Returns [N, D] float tensor where:
            N = number of vision tokens (varies with image resolution)
            D = SigLIP hidden dimension (1152 for Gemma 4)
        """
        image = image.convert("RGB")

        # Minimal prompt — the text "d" is a throwaway. We need
        # apply_chat_template to produce pixel_values and
        # image_position_ids in the correct format for the vision encoder.
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "d"},
        ]}]
        inputs = self.processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            vis = self.model.model.get_image_features(
                inputs["pixel_values"].to(self.device),
                inputs.get("image_position_ids", None),
                return_dict=True,
            )
            features = vis.pooler_output
            # SigLIP may return [1, N, D] or [N, D] depending on batch
            if features.dim() == 3:
                features = features.squeeze(0)

        return features

    def native_token_count(self, image: Image.Image) -> int:
        """Count native vision tokens without running the full vision encoder.

        Processes the image through the chat template to get input_ids,
        then counts image_token placeholders between BOI and EOI markers.
        Cheaper than extract() when you only need the count.
        """
        image = image.convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "d"},
        ]}]
        inputs = self.processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt"
        )
        ids = inputs["input_ids"][0].tolist()

        boi = self.model.config.boi_token_id
        eoi = self.model.config.eoi_token_id
        bp = ids.index(boi)
        ep = ids.index(eoi)
        return ep - bp - 1

    def generate_stock(self, image: Image.Image, prompt: str,
                       gen_config: Optional[dict] = None) -> str:
        """Standard Gemma 4 inference — no compression."""
        image = image.convert("RGB")
        gen = {**GENERATION_DEFAULTS, **(gen_config or {})}

        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen)

        raw = self.tokenizer.decode(output[0], skip_special_tokens=False)
        answer, _ = parse_gemma4_output(raw)
        return answer

    def generate_compressed(self, image: Image.Image, prompt: str,
                            features: torch.Tensor,
                            gen_config: Optional[dict] = None) -> str:
        """Inject compressed vision features and generate.

        Mechanism (monkey-patch):
          1. Build input_ids with M image-token placeholders (M = len(features))
          2. Replace model.model.get_image_features with a function that
             returns our pre-compressed features instead of running SigLIP
          3. Provide dummy pixel_values to trigger the vision code path
          4. Generate
          5. Restore the original get_image_features (always, even on error)

        This approach works across all Gemma 4 sizes because it operates
        above the vision encoder — the LLM sees features in the same format
        regardless of how they were produced.
        """
        image = image.convert("RGB")
        gen = {**GENERATION_DEFAULTS, **(gen_config or {})}
        n_compressed = len(features)

        # Read special token IDs from model config (not hardcoded).
        # These are the same across Gemma 4 sizes but reading from config
        # is safer for forward compatibility.
        boi = self.model.config.boi_token_id
        eoi = self.model.config.eoi_token_id
        img_tok = self.model.config.image_token_id

        # Step 1: Get the original input_ids from the processor, then
        # rebuild with the compressed token count.
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        processed = self.processor.apply_chat_template(
            msgs, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        ids = processed["input_ids"][0].tolist()

        bp = ids.index(boi)
        ep = ids.index(eoi)
        # Replace the native image tokens with our compressed count
        new_ids = ids[:bp] + [boi] + [img_tok] * n_compressed + [eoi] + ids[ep + 1:]
        input_ids = torch.tensor([new_ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)

        # Step 2: Prepare features for injection.
        # Cast to model dtype (bfloat16) to match the LLM's embedding space.
        bound_features = features.clone().to(
            dtype=next(self.model.parameters()).dtype,
            device=self.device,
        )

        # Step 3: Monkey-patch the vision encoder.
        original_fn = self.model.model.get_image_features

        def inject(*args, _feat=bound_features, **kwargs):
            return _FakeVisionOutput(_feat)

        self.model.model.get_image_features = inject

        # Step 4: Create dummy pixel_values.
        # Shape: (batch=1, n_tokens * 9, 3 * 14 * 14)
        #
        # WHY this specific shape:
        # Gemma 4 uses SigLIP which processes images as 14x14 patches
        # with 3 color channels. The factor of 9 comes from SigLIP's
        # internal 3x3 sub-tile arrangement per vision token.
        # These values are never read (our monkey-patch intercepts first),
        # but HuggingFace's generate() validates tensor shapes before
        # calling the vision encoder, so dimensions must be plausible.
        dummy_pv = torch.zeros(
            1, n_compressed * 9, 3 * 14 * 14,
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )

        try:
            self.model.eval()
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=dummy_pv,
                    **gen,
                )
        finally:
            # Step 5: ALWAYS restore, even if generate() throws.
            # Failing to restore corrupts the model for subsequent calls.
            self.model.model.get_image_features = original_fn

        raw = self.tokenizer.decode(output[0], skip_special_tokens=False)
        answer, _ = parse_gemma4_output(raw)
        return answer
