"""Micro-reflection consolidation for Echo agent corrections.

Turns user corrections into structured feedback memory entries.
"""

from pathlib import Path
from typing import Dict, Any
import httpx
from loguru import logger

from hermes_cli.agents.echo.memory import MemoryStore

# v0.3.1 (axis-D fence into the learning loop): scan the constructed messages
# before the consolidation Ollama call (the call_llm fence previously stopped
# at the main loop). No-op in the public build (anima safety package absent ->
# _PROMPT_GUARD is None); a raise is caught by the surrounding try -> best-effort
# skip (returns False), never blocks the user.
try:
    from anima.safety.prompt_guard import DEFAULT_PROMPT_GUARD as _PROMPT_GUARD
except ImportError:
    _PROMPT_GUARD = None


def build_reflection_prompt(correction_context: dict) -> str:
    """Format the micro-reflection prompt for Ollama."""
    user_msg = correction_context.get("user_msg", "")
    prior_response = correction_context.get("prior_response", "")

    return f"""The user corrected the assistant.

User correction: "{user_msg}"
What the assistant did before: "{prior_response}"

Extract the lesson from this exchange and format as a memory entry.
Be thorough -- include context so this is useful weeks from now.

Return:
---
name: [short-kebab-case-slug]
description: [one-line summary]
metadata:
  type: feedback
---

[The rule or lesson]
**Why:** [the reason the user gave, if any]
**How to apply:** [when and where this guidance kicks in]"""


def parse_reflection_response(response: str) -> dict:
    """Parse the LLM reflection response into a memory entry dict."""
    try:
        import yaml
    except ImportError:
        yaml = None

    parts = response.split("---", 2)
    if len(parts) >= 3 and yaml:
        try:
            frontmatter = yaml.safe_load(parts[1].strip()) or {}
        except Exception:
            frontmatter = {}
        body = parts[2].strip()
    else:
        frontmatter = {}
        body = response.strip()

    return {
        "name": frontmatter.get("name", "correction"),
        "description": frontmatter.get("description", ""),
        "type": "feedback",
        "content": body,
    }


def consolidate_correction(store: MemoryStore, context: dict, ollama_config: dict) -> bool:
    """Run micro-reflection via Ollama and save to memory.

    Returns True on success, False on failure.
    """
    try:
        prompt = build_reflection_prompt(context)
        payload = {
            "model": ollama_config.get("model", "qwen3.6:35b"),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512,
            },
        }

        if _PROMPT_GUARD is not None:
            _PROMPT_GUARD.assert_messages_clean(payload["messages"])

        response = httpx.post(
            ollama_config.get("api_url", "http://localhost:11434/api/chat"),
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
        content = result["message"]["content"]

        parsed = parse_reflection_response(content)

        store.write(
            name=parsed["name"],
            description=parsed["description"],
            content=parsed["content"],
            mem_type="feedback",
        )

        logger.info(f"Correction reflection saved: {parsed['name']}")
        return True

    except Exception as e:
        logger.error(f"Correction reflection failed: {e}")
        return False
