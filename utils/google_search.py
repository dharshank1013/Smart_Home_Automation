"""
utils/google_search.py
Free web search — NO API KEY required.

Sources used (all free, no auth):
  1. Wikipedia REST API       — best for definitions / factual questions
  2. DuckDuckGo Instant API   — quick answers, zero-click info
  3. DuckDuckGo HTML scrape   — fallback web results
"""

import re
import requests
from utils.logger import get_logger

logger = get_logger(__name__)

HEADERS = {
    "User-Agent": "SmartHomeAI/1.0 (research project; python-requests)"
}

WIKI_TRIGGERS = [
    "what is", "what are", "who is", "who was", "what was",
    "define", "meaning of", "explain", "tell me about",
    "encyclopedia", "history of", "biography",
]


class GoogleSearchClient:
    """
    Drop-in replacement — no API key needed at all.
    update() kept for compatibility but does nothing.
    """

    def __init__(self):
        pass

    def update(self, api_key: str = "", cx: str = ""):
        pass  # no credentials needed

    # ── public entry point ────────────────────────────────────────────────────

    def search(self, query: str, num: int = 3) -> str:
        query = query.strip()
        if not query:
            return "Please give me something to search for."

        q_lower = query.lower()

        # Wikipedia first for factual / definition queries
        if any(t in q_lower for t in WIKI_TRIGGERS):
            result = self._wikipedia(query)
            if result:
                return result

        # DuckDuckGo Instant Answer
        result = self._ddg_instant(query)
        if result:
            return result

        # DuckDuckGo HTML scrape fallback
        result = self._ddg_scrape(query)
        if result:
            return result

        return f"Sorry, I could not find anything for **'{query}'**. Try rephrasing."

    # ── Source 1: Wikipedia REST API ─────────────────────────────────────────

    def _wikipedia(self, query: str) -> str:
        try:
            # Find best matching page title
            search_url = "https://en.wikipedia.org/w/api.php"
            params = {
                "action":   "query",
                "list":     "search",
                "srsearch": query,
                "format":   "json",
                "srlimit":  1,
            }
            resp = requests.get(search_url, params=params, headers=HEADERS, timeout=8)
            resp.raise_for_status()
            results = resp.json().get("query", {}).get("search", [])
            if not results:
                return ""

            title = results[0]["title"]

            # Fetch summary for that page
            summary_url = (
                f"https://en.wikipedia.org/api/rest_v1/page/summary/"
                f"{requests.utils.quote(title)}"
            )
            resp2 = requests.get(summary_url, headers=HEADERS, timeout=8)
            resp2.raise_for_status()
            data = resp2.json()

            page_title = data.get("title", title)
            extract    = data.get("extract", "")
            wiki_url   = data.get("content_urls", {}).get("desktop", {}).get("page", "")

            if not extract:
                return ""

            # Trim to 3 sentences
            sentences = re.split(r'(?<=[.!?])\s+', extract)
            summary   = " ".join(sentences[:3])

            lines = [f"📖 **{page_title}** — Wikipedia\n", summary]
            if wiki_url:
                lines.append(f"\n🔗 Read more: {wiki_url}")
            return "\n".join(lines)

        except Exception as e:
            logger.warning(f"Wikipedia search failed: {e}")
            return ""

    # ── Source 2: DuckDuckGo Instant Answer API ───────────────────────────────

    def _ddg_instant(self, query: str) -> str:
        try:
            url    = "https://api.duckduckgo.com/"
            params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
            resp   = requests.get(url, params=params, headers=HEADERS, timeout=8)
            resp.raise_for_status()
            data = resp.json()

            abstract   = data.get("AbstractText", "").strip()
            source     = data.get("AbstractSource", "")
            abs_url    = data.get("AbstractURL", "")
            answer     = data.get("Answer", "").strip()
            definition = data.get("Definition", "").strip()
            def_source = data.get("DefinitionSource", "")
            def_url    = data.get("DefinitionURL", "")

            lines = []
            if answer:
                lines.append(f"💡 **Answer:** {answer}")
            if abstract:
                lines.append(f"📝 **{source or 'DuckDuckGo'}:**\n{abstract}")
                if abs_url:
                    lines.append(f"🔗 {abs_url}")
            if definition and not abstract:
                lines.append(f"📖 **Definition ({def_source}):**\n{definition}")
                if def_url:
                    lines.append(f"🔗 {def_url}")

            # Add up to 2 related topics
            shown = 0
            for t in data.get("RelatedTopics", []):
                if shown >= 2:
                    break
                if isinstance(t, dict) and t.get("Text"):
                    lines.append(f"• {t['Text'][:120]}")
                    shown += 1

            return "\n".join(lines) if lines else ""

        except Exception as e:
            logger.warning(f"DuckDuckGo instant failed: {e}")
            return ""

    # ── Source 3: DuckDuckGo HTML scrape fallback ─────────────────────────────

    def _ddg_scrape(self, query: str) -> str:
        try:
            url  = "https://html.duckduckgo.com/html/"
            resp = requests.post(url, data={"q": query}, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            html = resp.text

            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', html, re.DOTALL)
            titles   = re.findall(r'class="result__a"[^>]*>(.*?)</a>',       html, re.DOTALL)

            def strip_tags(s):
                return re.sub(r'<[^>]+>', '', s).strip()

            titles   = [strip_tags(t) for t in titles[:3]]
            snippets = [strip_tags(s) for s in snippets[:3]]

            if not snippets:
                return ""

            lines = [f"🔍 Search results for **{query}**:\n"]
            for i, (title, snippet) in enumerate(zip(titles, snippets), 1):
                if title and snippet:
                    lines.append(f"**{i}. {title}**\n{snippet}\n")

            return "\n".join(lines) if len(lines) > 1 else ""

        except Exception as e:
            logger.warning(f"DuckDuckGo scrape failed: {e}")
            return ""
