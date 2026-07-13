"""Echo agent state definition for LangGraph"""

from typing import TypedDict, List, Dict, Any, Optional


class Message(TypedDict, total=False):
    """A conversation message"""
    role: str
    content: str
    name: Optional[str]


class ToolCall(TypedDict, total=False):
    """A parsed tool call from the LLM response"""
    name: str
    parameters: Dict[str, Any]


class ToolResult(TypedDict, total=False):
    """Result of a tool execution"""
    name: str
    success: bool
    output: str
    error: Optional[str]


class EchoState(TypedDict, total=False):
    """LangGraph state for the Echo agent"""

    user_input: str
    config: Dict[str, Any]

    messages: List[Message]

    pending_tool_calls: List[ToolCall]
    tool_results: List[ToolResult]

    memory_context: List[str]

    iteration_count: int
    workspace: str
    response: Optional[str]
    correction_context: Optional[Dict[str, str]]  # {user_msg, prior_response} or None
    pending_memory_fact: Optional[str]  # Queued fact for auto-memory consolidation
    pending_idea: Optional[str]  # Raw idea text from /idea command
    idea_active: bool  # Whether agent is in ideation exploration mode
    idea_start_index: Optional[int]  # Message index where /idea exploration began
    pending_session_action: Optional[str]  # "summarize" on /exit, else None
    auto_memory_count: int  # Tracks auto-memory saves this session
    should_continue: bool

    # --- Latin tutor module (--latin mode, 2026-07-12). Additive, optional;
    # only set when the graph is built with --latin. load_latin_state (the graph
    # node rewired between process_input and call_llm) reads the non-protected
    # latin ledger + sets latin_state; build_latin_system_prompt renders it.
    # translate_permitted is a per-turn /translate flag set in the echo_cmd REPL
    # loop (a user-typed override, NOT an LLM-decided tool param). Neither key
    # is a banned affect field (assert_state_keys_clean_for_prompt permits
    # them). Absent on plain (non-latin) graphs. ---
    latin_state: Optional[Dict[str, Any]]
    translate_permitted: Optional[bool]
