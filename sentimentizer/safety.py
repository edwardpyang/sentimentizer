"""Prompt-injection patterns used to filter web search results."""

from __future__ import annotations

import re

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    # Allow filler words: "ignore all of your previous instructions".
    re.compile(
        r"(?i)\b(ignore|disregard|forget)\s+(\w+\s+){0,4}?"
        r"(previous|above|prior|earlier|preceding|all)\s+instructions?"
    ),
    re.compile(r"(?i)you\s+are\s+now\s+a\s+"),
    re.compile(r"(?i)system\s*:\s*"),
    re.compile(r"(?i)<\s*/?\s*(system|assistant|user)\s*>"),
]
