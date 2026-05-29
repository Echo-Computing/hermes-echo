"""Correction, fact, and command detection for Echo agent learning.

All detection uses regex only -- zero LLM calls.
"""

import re
from typing import Optional, Tuple


# Correction patterns from the design spec
CORRECTION_PATTERNS = [
    (r"no,?\s+(use|try|do|go with|we should)", "correction_directive"),
    (r"don'?t\s|do not\s", "correction_negative"),
    (r"\binstead\b", "correction_instead"),
    (r"\bactually\b", "correction_actually"),
    (r"\bprefer\b", "correction_prefer"),
    (r"\balways\b|\bnever\b", "correction_rule"),
]

# Explicit fact triggers
EXPLICIT_FACT_PATTERNS = [
    r"\bremember\b",
    r"\bsave this\b",
    r"\bdon'?t forget\b",
    r"\bREMEMBER\b",
]

# Strong signal patterns
STRONG_SIGNAL_PATTERNS = [
    r"\bI prefer\b",
    r"\bmy setup\b",
    r"\blet'?s go with\b",
    r"\bthe plan is\b",
    r"\balways\b",
    r"\bnever\b",
]

# Command patterns
COMMAND_PATTERNS = {
    "idea_save": r"^/idea\s+save\s*$",
    "idea": r"^/idea\s+(.+)$",
    "exit": r"^/exit\s*$",
}


def detect_correction(user_msg: str) -> bool:
    """Check if a user message contains a correction pattern.

    Returns True if any correction pattern matches.
    """
    for pattern, _ in CORRECTION_PATTERNS:
        if re.search(pattern, user_msg, re.IGNORECASE):
            return True
    return False


def detect_fact(user_msg: str) -> Tuple[Optional[str], str]:
    """Detect memory-worthy facts in a user message.

    Returns (tier, matched_text) where tier is:
    - "explicit" for explicit saves (remember, save this, etc.)
    - "strong" for strong signals (I prefer, my setup, etc.)
    - None if no match
    """
    for pattern in EXPLICIT_FACT_PATTERNS:
        if re.search(pattern, user_msg):
            return "explicit", user_msg

    for pattern in STRONG_SIGNAL_PATTERNS:
        if re.search(pattern, user_msg, re.IGNORECASE):
            return "strong", user_msg

    return None, ""


def detect_command(user_msg: str) -> dict:
    """Detect special commands in user input.

    Returns dict with:
    - "command": "idea" | "idea_save" | "exit" | None
    - "arg": argument string or None
    """
    for cmd, pattern in COMMAND_PATTERNS.items():
        match = re.match(pattern, user_msg.strip())
        if match:
            if cmd == "idea":
                return {"command": "idea", "arg": match.group(1).strip()}
            elif cmd == "idea_save":
                return {"command": "idea_save", "arg": None}
            elif cmd == "exit":
                return {"command": "exit", "arg": None}

    return {"command": None, "arg": None}
