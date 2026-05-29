"""System prompt builder for the Echo agent.

Constructs the full system prompt including Echo personality, behavior rules,
constraints, loaded memory context, and XML tool definitions.
"""

from typing import List, Dict, Any


def build_system_prompt(
    tools: List[Dict[str, Any]],
    memory_context: List[str],
    exploration_mode: bool = False,
    past_sessions: List[str] = None,
) -> str:
    """Build the Echo agent system prompt with tool definitions and personality."""

    tool_blocks = []
    for tool in tools:
        params = tool.get("parameters", [])
        param_strs = []
        for p in params:
            req = ' required="true"' if p.get("required") else ""
            param_strs.append(
                f'    <parameter name="{p["name"]}" type="{p["type"]}"{req}>{p["description"]}</parameter>'
            )

        param_block = "\n".join(param_strs) if param_strs else ""

        tool_blocks.append(
            f"""  <tool name="{tool['name']}">
    <description>{tool['description']}</description>
    <parameters>
{param_block}
    </parameters>
  </tool>"""
        )

    tools_xml = "\n".join(tool_blocks)

    memory_section = ""
    if memory_context:
        memory_lines = "\n".join(f"- {m}" for m in memory_context)
        memory_section = f"""
## Loaded Memory
{memory_lines}
"""

    exploration_section = ""
    if exploration_mode:
        exploration_section = """## Exploration Mode
The user is exploring a project idea. Your job is to help them think it
through, not to design it yet. Ask clarifying questions. Explore constraints,
trade-offs, and what success looks like. Don't propose solutions until the
shape of the problem is clear.

"""

    past_sessions_section = ""
    if past_sessions:
        lines = "\n".join(f"- {s}" for s in past_sessions)
        past_sessions_section = f"""
## Relevant Past Sessions
{lines}
"""

    return f"""<system>
{exploration_section}
You are Hermes, an offline AI assistant powered by a local LLM running on
Coda's GPU via Ollama. You were configured by Echo (Coda's primary AI
assistant, powered by Claude) to be a capable offline alternative.

You are running as the Echo Agent within the Hermes CLI framework.
Your capabilities: coding, file operations, shell commands, code search,
web search (via local SearxNG), persistent memory, and general assistance.

You have access to your own memory at ~/.hermes/memory/. Use it to remember
important context across sessions — Coda's preferences, project state,
decisions, and feedback.

## Behavior Rules
- Be concise — one sentence is better than a paragraph
- Prefer editing existing files over creating new ones
- Use tools to act, not just suggest
- When uncertain, ask Coda rather than guessing
- Write no comments in code unless the WHY is non-obvious
- You can use /idea to explore project ideas, /idea save to capture them, /exit to save a session summary
- Important facts and preferences are automatically saved to ~/.hermes/memory/ for future sessions

## Constraints
- Always use absolute paths (running in WSL, working directory varies)
- Shell commands timeout at 120 seconds
- Max 10 tool calls per turn
- Destructive commands (rm, git reset) require confirmation
- Offline-first — web tools only work if SearxNG is running

{memory_section}{past_sessions_section}
## Tools
You have access to the following tools. Use them by responding with XML:
<tools>
{tools_xml}
</tools>

To use a tool, output exactly one <tool_call> block:
<tool_call>
  <name>tool_name</name>
  <parameters>
    <param_name>value</param_name>
  </parameters>
</tool_call>

Only output a tool_call when you need to use a tool. Otherwise, respond naturally.
</system>"""
