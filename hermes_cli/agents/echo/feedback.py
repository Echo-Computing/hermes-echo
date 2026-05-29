"""Feedback loop for the Echo agent.

Captures corrections and preferences from conversations, stores them as
feedback memory entries, and detects patterns across entries.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from hermes_cli.agents.echo.memory import MemoryStore


class FeedbackLoop:
    """Captures, stores, and summarizes feedback from conversations."""

    def __init__(self, memory_dir: Path):
        self.store = MemoryStore(memory_dir)

    @staticmethod
    def _extract_trigger(content: str) -> str:
        """Extract the trigger text from a feedback entry's content."""
        body = content.split("---", 2)[-1] if content and content.startswith("---") else content
        trigger_line = body.strip().split("\n")[0]
        return trigger_line.split("Trigger: '", 1)[-1].rstrip("'")

    def capture(self, user_input: str, assistant_response: str, context: Optional[str] = None) -> Optional[str]:
        """Detect if the user is giving feedback and capture it.

        Returns a confirmation message if feedback was captured, or None if no feedback detected.
        """
        lower = user_input.lower()
        feedback_patterns = [
            (r"don't (?:ever )?(?:do|use|make|write|run|try|put|add)", "avoid_pattern"),
            (r"(?:never|always) [sS]top [sS]topping", "avoid_pattern"),
            (r"don't ever [^\s]+", "avoid_pattern"),
            (r"don't use [^\s]+", "avoid_pattern"),
            (r"from now on i(?:'?m| am| prefer| want| use| try)", "from_now_on"),
            (r"use [^\s]+ instead", "preference"),
            (r"prefer [^\s]+", "preference"),
            (r"change my [^\s]+", "preference_update"),
            (r"actually i(?:'?m| am| want| need)", "preference_update"),
            (r"i (?:prefer|want|need) [^\s]+", "preference"),
            (r"note that i(?:'?m| am)", "note"),
            (r"remember that (?:i(?:'?m| am)|you)", "remember"),
        ]

        # Dedup check before writing
        existing = self.read_all_feedback()
        for entry in existing[-3:]:
            trigger_text = self._extract_trigger(entry["content"])
            if trigger_text.lower() == lower:
                return None  # Duplicate

        for pattern, ftype in feedback_patterns:
            try:
                if re.search(pattern, lower):
                    content = f"Trigger: '{user_input[:200]}'\nType: {ftype}"
                    if context:
                        content += f"\nContext: {context[:300]}"
                    content += f"\nTime: {datetime.now().isoformat()}"

                    self.store.write(
                        f"feedback-{datetime.now().strftime('%Y%m%d-%H%M%S%f')}",
                        f"User correction ({ftype})",
                        content,
                        "feedback",
                    )
                    return f"Feedback saved: {ftype}"
            except re.error:
                continue

        return None

    def read_all_feedback(self) -> List[Dict]:
        """Read all feedback entries."""
        feedbacks = []
        for md_file in self.store.memory_dir.rglob("*.md"):
            content = self.store.read(md_file.stem)
            if content and "Trigger:" in content:
                feedbacks.append({
                    "name": md_file.stem,
                    "content": content,
                })
        feedbacks.sort(key=lambda x: x["name"], reverse=True)
        return feedbacks

    def summarize_patterns(self) -> str:
        """Detect patterns across feedback entries and summarize."""
        feedbacks = self.read_all_feedback()
        if len(feedbacks) < 3:
            return "Not enough feedback to find patterns yet."

        type_counts = {}
        for fb in feedbacks:
            content = fb["content"]
            for line in content.split("\n"):
                if line.startswith("Type:"):
                    ftype = line.split(":", 1)[1].strip()
                    type_counts[ftype] = type_counts.get(ftype, 0) + 1

        patterns = []
        for ftype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            patterns.append(f"- {ftype}: {count} occurrence(s)")

        return "\n".join(patterns) if patterns else "No patterns detected."
