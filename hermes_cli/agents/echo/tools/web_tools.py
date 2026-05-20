"""Web tools for the Echo agent — search and fetch via local SearxNG.

These tools are offline-first: if SearxNG is not running, they report unavailability
rather than failing.
"""

import re
import httpx


def search_web(query: str, limit: int = 10, searxng_url: str = "http://localhost:8080") -> str:
    """Search the web using a local SearxNG instance.

    Args:
        query: The search query string.
        limit: Maximum number of results to return.
        searxng_url: Base URL of the SearxNG instance.

    Returns:
        Formatted search results or an error message.
    """
    try:
        response = httpx.get(
            f"{searxng_url}/search",
            params={"q": query, "format": "json", "language": "en"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])[:limit]

        if not results:
            return "No results found."

        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = (r.get("content", "") or "")[:200]
            lines.append(f"{i}. {title}\n   {url}\n   {snippet}...")

        return "\n\n".join(lines)
    except Exception as e:
        return f"Web search unavailable — is SearxNG Docker running? Error: {e}"


def fetch_url(url: str) -> str:
    """Fetch and extract text content from a URL.

    Args:
        url: The URL to fetch.

    Returns:
        Extracted text content (first 8000 characters) or an error message.
    """
    try:
        response = httpx.get(url, timeout=30, follow_redirects=True)
        response.raise_for_status()
        content = response.text

        # Strip HTML tags
        text = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text[:8000] if text else "(empty page)"
    except Exception as e:
        return f"Failed to fetch URL — is SearxNG Docker running? Error: {e}"
