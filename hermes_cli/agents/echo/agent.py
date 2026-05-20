"""Echo agent LangGraph definition.

Builds a 5-node state graph for the Echo chat agent:
  process_input -> call_llm -> router -> execute_tools -> call_llm (loop)
                              -> format_response (end)
"""

from pathlib import Path
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from loguru import logger

import httpx
import re

from hermes_cli.agents.echo.state import EchoState
from hermes_cli.agents.echo.memory import MemoryStore
from hermes_cli.agents.echo.system_prompt import build_system_prompt
from hermes_cli.agents.echo.tools.registry import ToolRegistry, Tool

# Tool function imports
from hermes_cli.agents.echo.tools.file_tools import read_file, write_file, edit_file
from hermes_cli.agents.echo.tools.shell_tools import run_shell
from hermes_cli.agents.echo.tools.search_tools import search_code
from hermes_cli.agents.echo.tools.memory_tools import MemoryTool
from hermes_cli.agents.echo.tools.web_tools import search_web, fetch_url


def _build_registry(memory_dir: Path) -> ToolRegistry:
    """Build and return a ToolRegistry with all Echo tools registered."""
    reg = ToolRegistry()

    reg.register(Tool(
        name="read_file",
        description="Read a file from the filesystem",
        parameters=[
            {"name": "path", "type": "string", "required": True, "description": "Absolute file path"},
            {"name": "offset", "type": "int", "required": False, "description": "Start line number"},
            {"name": "limit", "type": "int", "required": False, "description": "Max lines to read"},
        ],
        handler=read_file,
    ))
    reg.register(Tool(
        name="write_file",
        description="Create or overwrite a file",
        parameters=[
            {"name": "path", "type": "string", "required": True, "description": "Absolute file path"},
            {"name": "content", "type": "string", "required": True, "description": "File content to write"},
        ],
        handler=write_file,
    ))
    reg.register(Tool(
        name="edit_file",
        description="Replace a string in an existing file (first occurrence)",
        parameters=[
            {"name": "path", "type": "string", "required": True, "description": "Absolute file path"},
            {"name": "old_string", "type": "string", "required": True, "description": "Exact text to replace"},
            {"name": "new_string", "type": "string", "required": True, "description": "Replacement text"},
        ],
        handler=edit_file,
    ))
    reg.register(Tool(
        name="run_shell",
        description="Execute a shell command in WSL/Linux",
        parameters=[
            {"name": "command", "type": "string", "required": True, "description": "Shell command to execute"},
            {"name": "timeout", "type": "int", "required": False, "description": "Timeout in seconds"},
        ],
        handler=run_shell,
    ))
    reg.register(Tool(
        name="search_code",
        description="Search for files (glob) or content (grep) in a directory",
        parameters=[
            {"name": "pattern", "type": "string", "required": True, "description": "Search pattern"},
            {"name": "type", "type": "string", "required": False, "description": "Search type: glob or grep"},
            {"name": "path", "type": "string", "required": False, "description": "Base directory to search"},
        ],
        handler=search_code,
    ))
    reg.register(Tool(
        name="memory",
        description="Search, read, or write persistent memory in ~/.hermes/memory/",
        parameters=[
            {"name": "action", "type": "string", "required": True, "description": "One of: search, read, write"},
            {"name": "query", "type": "string", "required": False, "description": "Search query (for search action)"},
            {"name": "name", "type": "string", "required": False, "description": "Memory name (for read/write actions)"},
            {"name": "content", "type": "string", "required": False, "description": "Memory content (for write action)"},
            {"name": "description", "type": "string", "required": False, "description": "Memory description (for write action)"},
        ],
        handler=None,  # Handled specially in execute_tools
    ))
    reg.register(Tool(
        name="search_web",
        description="Search the internet via local SearxNG instance",
        parameters=[
            {"name": "query", "type": "string", "required": True, "description": "Search query"},
            {"name": "limit", "type": "int", "required": False, "description": "Max number of results"},
        ],
        handler=search_web,
    ))
    reg.register(Tool(
        name="fetch_url",
        description="Fetch and extract text content from a URL",
        parameters=[
            {"name": "url", "type": "string", "required": True, "description": "URL to fetch"},
        ],
        handler=fetch_url,
    ))

    return reg


def process_input(state: EchoState) -> EchoState:
    """Node 1: Process user input and load relevant memory context."""
    logger.info("Echo: processing input")

    memory_dir = Path(state["config"].get("memory_dir", str(Path.home() / ".hermes" / "memory")))
    store = MemoryStore(memory_dir)
    memory_results = store.search(state["user_input"])
    state["memory_context"] = [f"{r['name']}: {r['description']}" for r in memory_results[:3]]

    state["iteration_count"] = 0
    state["tool_results"] = []
    state["should_continue"] = True
    state["response"] = None
    return state


def call_llm(state: EchoState) -> EchoState:
    """Node 2: Send current state to Ollama and get the LLM response."""
    config = state["config"]
    api_url = config.get("api_url", "http://localhost:11434/api/chat")
    model = config.get("model", "qwen3.6:35b")
    max_tokens = config.get("max_tokens", 4096)
    temperature = config.get("temperature", 0.7)

    memory_dir = Path(config.get("memory_dir", str(Path.home() / ".hermes" / "memory")))
    registry = _build_registry(memory_dir)

    system_prompt = build_system_prompt(
        registry.list_tools(),
        state.get("memory_context", [])
    )

    # Build messages array
    messages = [{"role": "system", "content": system_prompt}]

    # Add conversation history (last N messages)
    context_limit = config.get("context_messages", 50)
    for msg in state.get("messages", [])[-context_limit:]:
        messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    # Add the current user input
    messages.append({"role": "user", "content": state["user_input"]})

    # Add tool results from previous iteration as context
    for tr in state.get("tool_results", []):
        tool_msg = f"Tool '{tr['name']}' completed.\nOutput:\n{tr['output']}"
        if tr.get("error"):
            tool_msg += f"\nError: {tr['error']}"
        messages.append({"role": "user", "content": tool_msg})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    logger.info(f"Echo: calling Ollama ({model})")

    try:
        response = httpx.post(api_url, json=payload, timeout=120.0)
        response.raise_for_status()
        result = response.json()
        content = result["message"]["content"]
        state["response"] = content
    except Exception as e:
        logger.error(f"Echo: Ollama call failed: {e}")
        state["response"] = f"Error calling model: {e}"
        state["should_continue"] = False

    return state


def router(state: EchoState) -> str:
    """Node 3: Check if the LLM response contains tool calls, and route accordingly."""
    if not state.get("should_continue", False):
        return "format_response"

    max_calls = state["config"].get("max_tool_calls", 10)
    if state.get("iteration_count", 0) >= max_calls:
        logger.warning(f"Echo: max tool calls ({max_calls}) reached")
        return "format_response"

    response = state.get("response", "")
    memory_dir = Path(state["config"].get("memory_dir", str(Path.home() / ".hermes" / "memory")))
    registry = _build_registry(memory_dir)

    if registry.has_tool_calls(response):
        logger.info("Echo: routing to execute_tools")
        return "execute_tools"

    logger.info("Echo: routing to format_response")
    return "format_response"


def execute_tools(state: EchoState) -> EchoState:
    """Node 4: Parse and execute tool calls from the LLM response."""
    memory_dir = Path(state["config"].get("memory_dir", str(Path.home() / ".hermes" / "memory")))
    registry = _build_registry(memory_dir)

    calls = registry.parse_tool_calls(state.get("response", ""))
    results = []

    for call in calls:
        tool_name = call["name"]
        params = call.get("parameters", {})

        # Special handling for memory tool (class-based, stateful)
        if tool_name == "memory":
            mem_tool = MemoryTool(memory_dir)
            action = params.get("action", "search")
            try:
                if action == "search":
                    output = mem_tool.search(params.get("query", ""))
                elif action == "read":
                    output = mem_tool.read(params.get("name", ""))
                elif action == "write":
                    output = mem_tool.write(
                        params.get("name", ""),
                        params.get("description", ""),
                        params.get("content", ""),
                        params.get("type", "reference"),
                    )
                else:
                    output = f"Unknown memory action: '{action}'. Use search, read, or write."
                results.append({"name": tool_name, "success": True, "output": output, "error": None})
            except Exception as e:
                results.append({"name": tool_name, "success": False, "output": "", "error": str(e)})
        else:
            # Standard tools via registry
            results.append(registry.execute(tool_name, params))

    state["tool_results"] = results
    state["iteration_count"] = state.get("iteration_count", 0) + 1

    # Append tool results as conversation messages
    for res in results:
        msg_content = f"Tool '{res['name']}' result:\n{res['output']}"
        if res.get("error"):
            msg_content += f"\nError: {res['error']}"
        state["messages"] = state.get("messages", []) + [{"role": "tool", "content": msg_content}]

    logger.info(f"Echo: executed {len(results)} tool(s), iteration {state['iteration_count']}")
    return state


def format_response(state: EchoState) -> EchoState:
    """Node 5: Clean XML artifacts from response and save to history."""
    response = state.get("response", "")

    # Strip tool_call and tools XML blocks from the final response
    cleaned = re.sub(
        r'<tool_call>.*?</tool_call>',
        '',
        response,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r'<tools>.*?</tools>',
        '',
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = cleaned.strip()

    state["response"] = cleaned
    state["should_continue"] = False

    # Save to conversation history
    state["messages"] = state.get("messages", []) + [
        {"role": "user", "content": state["user_input"]},
        {"role": "assistant", "content": cleaned},
    ]

    return state


def create_echo_graph():
    """Build and compile the Echo agent LangGraph.

    Returns a compiled StateGraph ready for graph.invoke(state).
    """
    workflow = StateGraph(EchoState)

    workflow.add_node("process_input", process_input)
    workflow.add_node("call_llm", call_llm)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("format_response", format_response)

    workflow.set_entry_point("process_input")
    workflow.add_edge("process_input", "call_llm")

    workflow.add_conditional_edges(
        "call_llm",
        router,
        {
            "execute_tools": "execute_tools",
            "format_response": "format_response",
        },
    )

    workflow.add_edge("execute_tools", "call_llm")
    workflow.add_edge("format_response", END)

    return workflow.compile()
