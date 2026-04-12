"""Tests for scoring and parsing utilities.

Pure string operations, no dependencies.
"""

import pytest

from vtbench.benchmark.scoring import soft_match, exact_match, evaluate_answer, extract_letter
from vtbench.models.gemma4.parsing import parse_gemma4_output


class TestSoftMatch:

    def test_gt_in_prediction(self):
        assert soft_match("a red car parked outside", "red car")

    def test_prediction_in_gt(self):
        assert soft_match("car", "a red car")

    def test_exact(self):
        assert soft_match("yes", "yes")

    def test_case_insensitive(self):
        assert soft_match("Yes", "yes")
        assert soft_match("RED CAR", "red car")

    def test_no_match(self):
        assert not soft_match("blue truck", "red car")

    def test_empty_prediction(self):
        assert not soft_match("", "red car")

    def test_empty_gt(self):
        assert not soft_match("red car", "")

    def test_both_empty(self):
        assert not soft_match("", "")

    def test_whitespace(self):
        assert soft_match("  yes  ", "yes")


class TestExactMatch:

    def test_match(self):
        assert exact_match("yes", "yes")

    def test_case_insensitive(self):
        assert exact_match("Yes", "yes")

    def test_no_match(self):
        assert not exact_match("no", "yes")

    def test_partial_no_match(self):
        """Partial overlap should NOT match for exact_match."""
        assert not exact_match("a red car", "red car")

    def test_empty(self):
        assert not exact_match("", "yes")


class TestEvaluateAnswer:
    """Tests for auto-detecting scoring mode from ground truth format."""

    def test_mc_exact_letter(self):
        """Single letter GT triggers letter extraction."""
        assert evaluate_answer("B", "B")
        assert evaluate_answer("A", "A")

    def test_mc_extracts_from_text(self):
        """Letter extraction works on verbose model output."""
        assert evaluate_answer("The answer is B", "B")
        assert evaluate_answer("(C)", "C")
        assert evaluate_answer("Answer: D", "D")

    def test_mc_wrong_letter(self):
        assert not evaluate_answer("A", "B")
        assert not evaluate_answer("The answer is C", "D")

    def test_mc_empty_prediction(self):
        assert not evaluate_answer("", "B")

    def test_mc_no_letter_in_prediction(self):
        """Model output with no extractable letter fails."""
        assert not evaluate_answer("I don't know", "B")

    def test_vqa_fallback(self):
        """Multi-word GT triggers soft_match, not letter extraction."""
        assert evaluate_answer("a red car", "red car")
        assert evaluate_answer("car", "a red car")

    def test_vqa_no_match(self):
        assert not evaluate_answer("blue truck", "red car")

    def test_both_empty(self):
        assert not evaluate_answer("", "")

    def test_single_word_gt_not_letter(self):
        """Single word GT that isn't A-D uses soft_match."""
        assert evaluate_answer("yes", "yes")
        assert not evaluate_answer("no", "yes")


class TestParseGemma4Output:

    def test_standard_format(self):
        raw = "some prefix<turn|>The answer is 42<turn|>"
        answer, valid = parse_gemma4_output(raw)
        assert answer == "The answer is 42"
        assert valid

    def test_multiple_turns(self):
        raw = "prefix<turn|>first response<turn|>second response<turn|>"
        answer, valid = parse_gemma4_output(raw)
        assert answer == "second response"
        assert valid

    def test_thinking_block_as_fallback(self):
        """When only a thinking block exists (no real answer), use it as fallback."""
        raw = "<turn|><|channel>thought Let me analyze this carefully and determine the answer<turn|>"
        answer, valid = parse_gemma4_output(raw)
        assert "analyze" in answer
        assert valid

    def test_answer_before_thinking(self):
        """Real answer should take priority over thinking block."""
        raw = "prefix<turn|>The actual answer<turn|><|channel>thought Long reasoning chain that is more than ten chars<turn|>"
        answer, valid = parse_gemma4_output(raw)
        assert answer == "The actual answer"
        assert valid

    def test_garbage_filtered(self):
        raw = "prefix<turn|><|image token stuff<turn|>real answer<turn|>"
        answer, valid = parse_gemma4_output(raw)
        assert answer == "real answer"
        assert valid

    def test_empty_output(self):
        raw = "<turn|><turn|>"
        answer, valid = parse_gemma4_output(raw)
        assert not valid

    def test_no_turn_markers(self):
        raw = "just some text without markers"
        answer, valid = parse_gemma4_output(raw)
        assert answer == "just some text without markers"
        assert valid


class TestExtractLetter:

    def test_answer_label(self):
        assert extract_letter("Answer: B") == "B"
        assert extract_letter("answer: (C)") == "C"

    def test_parenthesized(self):
        assert extract_letter("I think (A) is correct") == "A"

    def test_leading_letter(self):
        assert extract_letter("B. This is the answer") == "B"
        assert extract_letter("A) correct") == "A"

    def test_single_letter(self):
        assert extract_letter("D") == "D"

    def test_lowercase_converted(self):
        assert extract_letter("b") == "B"

    def test_no_letter(self):
        assert extract_letter("I don't know") == ""

    def test_empty(self):
        assert extract_letter("") == ""

    def test_letter_in_word(self):
        """Letters embedded in words should still find standalone matches."""
        result = extract_letter("The answer B seems right")
        assert result == "B"

    def test_extended_range(self):
        """E-J are valid for datasets with more than 4 options."""
        assert extract_letter("E") == "E"
        assert extract_letter("J") == "J"

    def test_out_of_range(self):
        """K, L, etc. beyond J should not match."""
        assert extract_letter("K") == ""
        assert extract_letter("Z") == ""
