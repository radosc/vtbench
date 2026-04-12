"""Gemma 4 output parsing.

Gemma 4 uses a chat format with <turn|> delimiters:
  <bos>...<turn|>model response<turn|>

Some Gemma 4 variants (E12B+) produce thinking blocks:
  <|channel>thought ... reasoning ... <turn|>actual answer<turn|>

This parser handles both, extracting the final usable text answer.
"""


def parse_gemma4_output(raw_text: str) -> tuple[str, bool]:
    """Parse raw Gemma 4 generation output into a clean answer.

    Handles:
      - Standard chat format with <turn|> delimiters
      - Thinking/reasoning blocks (<|channel>thought)
      - Garbage filtering (stray image tokens, BOS markers)

    Returns:
        (answer_text, is_valid) — is_valid is False if no usable
        answer could be extracted (empty output, all garbage, etc.)
    """
    parts = raw_text.split("<turn|>")

    # First pass: look for a real answer (non-thinking, non-garbage).
    # Walk backwards — the answer is typically the second-to-last segment.
    thinking_content = None
    for part in reversed(parts):
        part = part.strip()
        if not part:
            continue

        # Remember thinking content as fallback, but don't return it yet —
        # the actual answer takes priority over reasoning traces.
        if part.startswith("<|channel>thought"):
            content = part.replace("<|channel>thought", "").strip()
            if content and len(content) > 10 and thinking_content is None:
                thinking_content = content
            continue

        # Skip framework-generated garbage
        if "<|image" in part or "<bos>" in part or "<|turn" in part:
            continue

        return part, True

    # Fallback: if no clean answer found, use thinking content
    if thinking_content:
        return thinking_content, True

    return "", False
