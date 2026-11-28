"""
PHISHGUARD — Shared probability formatting helpers.

Preserves original probability values in memory while providing consistent
human-readable formatting across agent cards, verdict displays, and chart labels.
"""
import math
from typing import Any, Optional


def _to_valid_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = float(value)
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def format_probability(value: Any) -> str:
    """
    Format a probability in [0, 1] consistently across the UI:
      - Exact 0.0 -> "0.0000"
      - Exact 1.0 -> "1.0000"
      - Strictly between 0 and 0.0001 -> "<0.0001"
      - Strictly between 0.9999 and 1 -> ">0.9999"
      - Otherwise -> 4 decimal places
      - Invalid / None -> "—"
    """
    val = _to_valid_float(value)
    if val is None:
        return "—"
    if val == 0.0:
        return "0.0000"
    if val == 1.0:
        return "1.0000"
    if 0.0 < val < 0.0001:
        return "<0.0001"
    if 0.9999 < val < 1.0:
        return ">0.9999"
    return f"{val:.4f}"


def format_probability_full(value: Any) -> str:
    """
    Format a probability with full precision for tooltips and expandable
    technical details without altering the underlying float.
    """
    val = _to_valid_float(value)
    if val is None:
        return "—"
    if val == 0.0:
        return "0"
    if val == 1.0:
        return "1"
    return f"{val:.12g}"
