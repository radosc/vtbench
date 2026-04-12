"""Scoring utilities for benchmark evaluation.

Provides answer comparison functions for different benchmark types:
  - VQA (GQA): soft string matching
  - Multiple choice (MMMU-Pro): letter extraction + exact match
"""

import re

# Letters supported for multiple-choice extraction and scoring.
# Must be consistent between extract_letter() and evaluate_answer().
_MC_LETTERS = set("ABCDEFGHIJ")


def soft_match(predicted: str, ground_truth: str) -> bool:
    """Soft string match for VQA evaluation.

    Returns True if the ground truth appears in the prediction
    or vice versa (case-insensitive). This is the standard GQA
    evaluation approach — exact match is too strict for free-form VQA.

    Examples:
        soft_match("a red car", "red car")        -> True
        soft_match("car", "a red car")             -> True
        soft_match("blue truck", "red car")        -> False
    """
    if not predicted or not ground_truth:
        return False
    p = predicted.lower().strip()
    g = ground_truth.lower().strip()
    return g in p or p in g


def exact_match(predicted: str, ground_truth: str) -> bool:
    """Case-insensitive exact match."""
    if not predicted or not ground_truth:
        return False
    return predicted.lower().strip() == ground_truth.lower().strip()


def extract_letter(text: str) -> str:
    """Extract a multiple-choice answer letter from model output.

    Tries several patterns in order of specificity:
      1. "Answer: (B)" or "answer: B"  — explicit label, supports A-J
      2. Parenthesized letter "(C)"    — supports A-J
      3. Leading letter "A." or "A)"   — supports A-J
      4. Entire text is one letter     — supports A-J
      5. Any isolated A-D at a word boundary (conservative fallback)

    Supports letters A-J in explicit contexts (patterns 1-4).
    Falls back to A-D only for the ambiguous word-boundary match (pattern 5)
    to avoid false positives like "I" (pronoun) matching as a letter.

    Returns uppercase letter or empty string if not found.
    """
    text = text.strip()

    # Pattern 1: explicit "answer" label — full A-J range
    m = re.search(r'[Aa]nswer[:\s]*\(?([A-J])\)?', text)
    if m:
        return m.group(1).upper()

    # Pattern 2: parenthesized letter — full A-J range
    m = re.search(r'\(([A-J])\)', text)
    if m:
        return m.group(1).upper()

    # Pattern 3: leading letter with punctuation (not just whitespace —
    # "I don't know" starts with "I " which is a pronoun, not an answer)
    m = re.search(r'^([A-J])[.),:]', text)
    if m:
        return m.group(1).upper()

    # Pattern 4: entire response is a single letter — full A-J range
    if len(text) == 1 and text.upper() in _MC_LETTERS:
        return text.upper()

    # Pattern 5: any standalone A-D at word boundary (conservative —
    # limited to A-D to avoid false positives). Exclude matches followed
    # by apostrophe (catches "I don't", "A's", etc.)
    m = re.search(r'\b([A-D])\b(?!\')', text)
    if m:
        return m.group(1).upper()

    return ""


def evaluate_answer(predicted: str, ground_truth: str) -> bool:
    """Auto-detect scoring mode from ground truth format.

    If ground truth is a single letter in the supported MC range (A-J),
    use letter extraction. Otherwise use soft match.
    """
    if not predicted or not ground_truth:
        return False
    gt = ground_truth.strip().upper()
    if len(gt) == 1 and gt in _MC_LETTERS:
        return extract_letter(predicted) == gt
    return soft_match(predicted, ground_truth)
