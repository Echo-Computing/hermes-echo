"""Tool registry for the Echo agent.

Handles tool registration, XML parsing of <tool_call> blocks from LLM responses,
and tool execution with error handling.
"""

import re
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field


@dataclass
class Tool:
    """Definition of a tool the agent can use."""
    name: str
    description: str
    parameters: List[Dict[str, Any]]
    handler: Callable[..., Any]


class ToolRegistry:
    """Registry for tools — register, list, parse tool calls from text, and execute."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool in the registry."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return tool definitions for inclusion in the system prompt."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    def parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse <tool_call> XML blocks from LLM response text.

        Returns a list of dicts with 'name' and 'parameters' keys.
        """
        calls = []
        pattern = (
            r'<tool_call>\s*'
            r'<name>(.*?)</name>\s*'
            r'<parameters>(.*?)</parameters>\s*'
            r'</tool_call>'
        )

        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            name = match.group(1).strip()
            params_block = match.group(2)

            params = {}
            param_pattern = r'<(\w+)>(.*?)</\w+>'
            for pmatch in re.finditer(param_pattern, params_block, re.DOTALL):
                key = pmatch.group(1).strip()
                value = pmatch.group(2).strip()
                params[key] = value

            calls.append({"name": name, "parameters": params})

        return calls

    def has_tool_calls(self, text: str) -> bool:
        """Check if the text contains any <tool_call> blocks."""
        return bool(re.search(r'<tool_call>', text, re.IGNORECASE))

    def execute(self, name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name with given parameters.

        Returns a dict with name, success, output, and error keys.
        """
        tool = self.get(name)
        if not tool:
            return {
                "name": name,
                "success": False,
                "output": "",
                "error": f"Tool '{name}' not found",
            }

        try:
            result = tool.handler(**parameters)
            return {
                "name": name,
                "success": True,
                "output": str(result) if result is not None else "",
                "error": None,
            }
        except Exception as e:
            return {
                "name": name,
                "success": False,
                "output": "",
                "error": str(e),
            }
