"""Session summary for Echo agent.

Writes structured session records on /exit and extracts individual learnings.
"""

from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any
import httpx
from loguru import logger

from hermes_cli.agents.echo.memory import MemoryStore

# v0.3.1 (integrity guard fence into the learning loop): scan the constructed
# messages before the consolidation Ollama call (the call_llm fence previously
# stopped at the main loop). No-op in the public build (safety package absent ->
# _PROMPT_GUARD is None); a raise is caught by the surrounding try -> best-effort
# skip (returns False), never blocks the user.
_PROMPT_GUARD = None


def build_summary_prompt(transcript: str) -> str:
    """Format the session summary prompt for Ollama."""
    return f"""Synthesize this conversation session into a structured record. Be thorough —
this is the artifact someone will read to understand what happened.

Session transcript:
{transcript}

Return:
## Summary
[What happened and what was accomplished]

## Key topics
- [topic]
...

## Decisions made
- [decision and the reasoning behind it]
...

## Corrections & learnings
- [what the assistant got wrong and what the correct approach is]
...

## New facts about the user
- [preferences, setup details, project context revealed]
...

## Carry-forward
[What the next session needs to know to continue where this one left off]"""


def parse_summary_response(response: str) -> dict:
    """Parse the LLM session summary into a sections dict."""
    sections = {
        "summary": "",
        "key_topics": [],
        "decisions": [],
        "corrections": [],
        "new_facts": [],
        "carry_forward": "",
    }

    current_section = None
    current_items = []

    for line in response.split("\n"):
        stripped = line.strip()
        if stripped.startswith("## "):
            # Save previous section
            if current_section:
                if current_section in ("summary", "carry_forward"):
                    sections[current_section] = "\n".join(current_items).strip()
                else:
                    sections[current_section] = [item for item in current_items if item]

            section_name = stripped[3:].lower().replace(" ", "_").replace("&", "and")
            if "summary" in section_name:
                current_section = "summary"
            elif "topic" in section_name:
                current_section = "key_topics"
            elif "decision" in section_name:
                current_section = "decisions"
            elif "correction" in section_name:
                current_section = "corrections"
            elif "fact" in section_name or "user" in section_name:
                current_section = "new_facts"
            elif "carry" in section_name:
                current_section = "carry_forward"
            else:
                current_section = None
            current_items = []
        elif current_section:
            if stripped:
                if current_section == "summary":
                    # Summary is free-form text — capture all non-empty lines
                    current_items.append(stripped)
                elif current_section == "carry_forward":
                    # Carry-forward is also free-form text
                    current_items.append(stripped)
                elif stripped.startswith("- "):
                    # Other sections are bulleted lists
                    current_items.append(stripped[2:].strip())

    # Save the last section
    if current_section:
        if current_section in ("summary", "carry_forward"):
            sections[current_section] = "\n".join(current_items).strip()
        else:
            sections[current_section] = [item for item in current_items if item]

    return sections


def write_session_record(history_dir: Path, transcript: str, summary: dict) -> Path:
    """Write session record to JSONL file. First line is summary, followed by transcript.

    Returns the path to the written file.
    """
    history_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    file_path = history_dir / f"{timestamp}.jsonl"

    summary_entry = {
        "date": timestamp,
        "type": "summary",
        **summary,
    }

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")
        f.write(json.dumps({"type": "transcript", "content": transcript}, ensure_ascii=False) + "\n")

    logger.info(f"Session record written: {file_path}")
    return file_path


def extract_learnings(store: MemoryStore, summary: dict) -> int:
    """Extract individual learnings from session summary and save to memory.

    Corrections -> feedback/
    Facts -> Knowledge/
    Decisions -> project file

    Returns count of new entries.
    """
    count = 0

    for correction in summary.get("corrections", []):
        if correction:
            store.write(
                name=f"session-correction-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                description=correction[:100],
                content=correction,
                mem_type="feedback",
            )
            count += 1

    for fact in summary.get("new_facts", []):
        if fact:
            store.write(
                name=f"session-fact-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                description=fact[:100],
                content=fact,
                mem_type="knowledge",
            )
            count += 1

    return count


def consolidate_session(store: MemoryStore, history_dir: Path, transcript: str, ollama_config: dict) -> bool:
    """Run session summary consolidation via Ollama and save to history + memory.

    Returns True on success, False on failure.
    """
    try:
        prompt = build_summary_prompt(transcript)
        payload = {
            "model": ollama_config.get("model", "qwen3.6:35b"),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 1536,
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

        parsed = parse_summary_response(content)
        write_session_record(history_dir, transcript, parsed)
        count = extract_learnings(store, parsed)

        logger.info(f"Session summary saved with {count} learning(s) extracted")
        return True

    except Exception as e:
        logger.error(f"Session summary failed: {e}")
        return False
