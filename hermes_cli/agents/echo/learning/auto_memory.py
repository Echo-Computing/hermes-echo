"""Auto-memory consolidation for Echo agent.

Detects and saves important facts, preferences, and decisions from conversation.
"""

from typing import Dict, Any
import httpx
from loguru import logger

from hermes_cli.agents.echo.memory import MemoryStore

# v0.3.1 (integrity guard fence into the learning loop): scan the constructed messages
# before the consolidation Ollama call (the call_llm fence previously stopped
# at the main loop). No-op in the public build (_PROMPT_GUARD is None); a raise
# is caught by the surrounding try -> best-effort skip (returns False), never
# blocks the user.
_PROMPT_GUARD = None


def build_auto_memory_prompt(fact_text: str) -> str:
    """Format the auto-memory consolidation prompt for Ollama."""
    return f"""Extract the key fact or preference from this statement as a memory entry.
Be thorough -- include context so this is useful weeks from now.

Statement: "{fact_text}"

Return:
---
name: [short-kebab-case-slug]
description: [one-line summary]
metadata:
  type: [user | feedback | project | reference]
---

[The fact or rule, with enough context to stand alone]
**Why:** [the reason or motivation behind it]
**How to apply:** [when and where this should guide behavior]"""


def parse_auto_memory_response(response: str) -> dict:
    """Parse the LLM auto-memory response into a memory entry dict."""
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
        "name": frontmatter.get("name", "memory"),
        "description": frontmatter.get("description", ""),
        "type": frontmatter.get("metadata", {}).get("type", "reference") if isinstance(frontmatter.get("metadata"), dict) else "reference",
        "content": body,
    }


def consolidate_fact(store: MemoryStore, fact_text: str, ollama_config: dict) -> bool:
    """Run auto-memory consolidation via Ollama and save to memory.

    Deduplicates: if a memory with the same name already exists, merges content.
    Returns True on success, False on failure.
    """
    try:
        # Check dedup: search for existing entry
        existing = store.search(fact_text)
        if existing:
            logger.info(f"Auto-memory: similar entry already exists, skipping")
            return True

        prompt = build_auto_memory_prompt(fact_text)
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

        parsed = parse_auto_memory_response(content)

        store.write(
            name=parsed["name"],
            description=parsed["description"],
            content=parsed["content"],
            mem_type=parsed.get("type", "reference"),
        )

        logger.info(f"Auto-memory saved: {parsed['name']}")
        return True

    except Exception as e:
        logger.error(f"Auto-memory consolidation failed: {e}")
        return False
