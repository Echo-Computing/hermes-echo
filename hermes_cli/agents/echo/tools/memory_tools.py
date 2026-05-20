"""Memory tool for the Echo agent — search, read, and write persistent memory."""

from pathlib import Path
from hermes_cli.agents.echo.memory import MemoryStore


class MemoryTool:
    """Memory tool wrapping MemoryStore for the Echo agent."""

    def __init__(self, memory_dir: Path):
        self.store = MemoryStore(memory_dir)

    def search(self, query: str) -> str:
        """Search memory for a query string."""
        results = self.store.search(query)
        if not results:
            return "No matching memories found."
        lines = [f"- {r['name']}: {r['description']} ({r['file']})" for r in results]
        return "\n".join(lines)

    def read(self, name: str) -> str:
        """Read a memory entry by name."""
        content = self.store.read(name)
        if content is None:
            return f"Memory '{name}' not found."
        return content

    def write(self, name: str, description: str, content: str, mem_type: str = "reference") -> str:
        """Write or update a memory entry."""
        self.store.write(name, description, content, mem_type)
        return f"Memory '{name}' saved."
