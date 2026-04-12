"""Gemma 4 model backend.

Supports all Gemma 4 variants: E2B-it, E4B-it, E12B-it, E27B-it.
"""

from vtbench.models.gemma4.backend import Gemma4Backend

# The discovery system looks for this exact name.
Backend = Gemma4Backend
