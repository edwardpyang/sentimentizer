"""Prompt-injection patterns used to filter web search results."""

from __future__ import annotations

import re

INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)ignore\s+(previous|above|prior)\s+instructions"),
    re.compile(r"(?i)forget\s+(previous|above|prior|all)\s+instructions"),
    re.compile(r"(?i)you\s+are\s+now\s+a\s+"),
    re.compile(r"(?i)system\s*:\s*"),
    re.compile(r"<\s*/?\s*(system|assistant|user)\s*>"),
]
