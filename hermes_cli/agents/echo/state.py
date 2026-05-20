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
    should_continue: bool
