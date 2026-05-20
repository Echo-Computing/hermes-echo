"""Persistent memory store for the Echo agent.

Uses the same frontmatter + MEMORY.md index format as Echo's Obsidian vault.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger


class MemoryStore:
    """Echo agent memory store — frontmatter + MEMORY.md format"""

    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.memory_dir / "MEMORY.md"

    def _parse_frontmatter(self, content: str) -> tuple:
        """Parse YAML frontmatter from a memory file. Returns (frontmatter_dict, body_text)."""
        if not content.startswith("---"):
            return {}, content

        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content

        try:
            import yaml
            frontmatter = yaml.safe_load(parts[1].strip()) or {}
        except Exception:
            frontmatter = {}

        return frontmatter, parts[2].strip()

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search MEMORY.md index and file descriptions for a query."""
        results = []
        query_lower = query.lower()

        for md_file in self.memory_dir.rglob("*.md"):
            if md_file.name == "MEMORY.md":
                continue

            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            frontmatter, body = self._parse_frontmatter(content)
            desc = frontmatter.get("description", "")
            name = frontmatter.get("name", md_file.stem)

            text = f"{name} {desc} {body[:500]}".lower()
            if query_lower in text:
                results.append({
                    "name": name,
                    "file": str(md_file.relative_to(self.memory_dir)),
                    "description": desc,
                })

        return results

    def read(self, name: str) -> Optional[str]:
        """Read a memory file by name or filename."""
        for md_file in self.memory_dir.rglob("*.md"):
            if md_file.name == "MEMORY.md":
                continue

            try:
                content = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            frontmatter, _ = self._parse_frontmatter(content)
            fm_name = frontmatter.get("name", md_file.stem)

            if fm_name == name or md_file.stem == name:
                return content

        return None

    def write(self, name: str, description: str, content: str, mem_type: str = "reference") -> None:
        """Create or update a memory file and update the MEMORY.md index."""
        safe_name = re.sub(r"[^\w\-]", "_", name).lower()
        file_path = self.memory_dir / f"{safe_name}.md"

        try:
            import yaml
        except ImportError:
            import json
            frontmatter_str = f"name: {name}\ndescription: {description}\n"
        else:
            frontmatter = {
                "name": name,
                "description": description,
                "metadata": {"type": mem_type},
            }
            frontmatter_str = yaml.dump(frontmatter, allow_unicode=True, default_flow_style=False)

        full_content = f"---\n{frontmatter_str}---\n\n{content}\n"
        file_path.write_text(full_content, encoding="utf-8")

        self._update_index(name, description, file_path)
        logger.info(f"Memory written: {file_path}")

    def _update_index(self, name: str, description: str, file_path: Path) -> None:
        """Update MEMORY.md index with a new or updated entry."""
        if not self.index_path.exists():
            self.index_path.write_text("# Memory Index\n\n", encoding="utf-8")

        index_content = self.index_path.read_text(encoding="utf-8")
        rel_path = file_path.relative_to(self.memory_dir)
        line = f"- [{name}]({rel_path}) — {description}\n"

        pattern = rf"- \[{re.escape(name)}\].*\n"
        if re.search(pattern, index_content):
            index_content = re.sub(pattern, line, index_content)
        else:
            index_content += line

        self.index_path.write_text(index_content, encoding="utf-8")
