"""Idea capture for Echo agent.

Two-stage: /idea enters exploration mode (0 LLM), /idea save extracts structured concept (1 LLM).
"""

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


EXPLORATION_PROMPT = """The user is exploring a project idea. Your job is to help them think it through, not to design it yet. Ask clarifying questions. Explore constraints, trade-offs, and what success looks like. Don't propose solutions until the shape of the problem is clear."""


def build_exploration_prompt() -> str:
    """Return the exploration mode system prompt augmentation."""
    return EXPLORATION_PROMPT


def build_idea_save_prompt(transcript: str) -> str:
    """Format the idea save consolidation prompt for Ollama."""
    return f"""Extract a structured project concept from this discussion. Be thorough --
capture everything that was explored so it's useful weeks from now.

Discussion:
{transcript}

Return a memory entry:
---
name: [short-kebab-slug]
description: [one-line hook summarizing the concept]
metadata:
  type: project
---

**Goal:** [what this project accomplishes]
**Context:** [why this project, what led to it]
**Constraints:** [limitations, requirements, dependencies discussed]
**Approaches explored:** [each approach with its trade-offs]
**Decisions made:** [what was settled, what's still open]
**Open questions:** [what still needs resolution before building]"""


def parse_idea_response(response: str) -> dict:
    """Parse the LLM idea save response into a memory entry dict."""
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
        "name": frontmatter.get("name", "project-idea"),
        "description": frontmatter.get("description", ""),
        "type": "project",
        "content": body,
    }


def consolidate_idea(store: MemoryStore, transcript: str, ollama_config: dict) -> bool:
    """Run idea save consolidation via Ollama and save to memory/projects/.

    Returns True on success, False on failure.
    """
    try:
        prompt = build_idea_save_prompt(transcript)
        payload = {
            "model": ollama_config.get("model", "qwen3.6:35b"),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1024,
            },
        }

        if _PROMPT_GUARD is not None:
            _PROMPT_GUARD.assert_messages_clean(payload["messages"])

        response = httpx.post(
            ollama_config.get("api_url", "http://localhost:11434/api/chat"),
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()
        content = result["message"]["content"]

        parsed = parse_idea_response(content)

        # Save to projects subdirectory
        project_dir = store.memory_dir / "projects"
        project_dir.mkdir(parents=True, exist_ok=True)
        project_store = MemoryStore(project_dir)
        project_store.write(
            name=parsed["name"],
            description=parsed["description"],
            content=parsed["content"],
            mem_type="project",
        )

        logger.info(f"Idea saved: {parsed['name']}")
        return True

    except Exception as e:
        logger.error(f"Idea save failed: {e}")
        return False
