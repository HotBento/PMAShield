"""
Filter research_tools_original.json, keeping only tools that can perform
academic paper search (arXiv, PubMed, Semantic Scholar, Google Scholar, etc.).
"""

import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Keywords that *strongly* indicate paper / academic search capability.
# Checked case-insensitively against the concatenation of tool name + description.
# ---------------------------------------------------------------------------
PAPER_KEYWORDS = [
    # Specific academic repositories / databases
    "arxiv",
    "pubmed",
    "biorxiv",
    "medrxiv",
    "iacr",
    "crossref",
    "semantic scholar",
    "semanticscholar",
    "google scholar",
    "google_scholar",
    "unpaywall",
    "openalex",
    "openaire",
    "core.ac.uk",
    "europepmc",
    "europe pmc",
    "scopus",
    "web of science",
    "ieee xplore",
    "acm digital library",
    # Generic paper-search phrases
    "paper search",
    "paper_search",
    "paper-search",
    "search paper",
    "search_paper",
    "academic paper",
    "research paper",
    "scientific paper",
    "scholarly paper",
    "academic article",
    "research article",
    "scientific article",
    "journal article",
    "preprint",
    "scholarly",
    "academic journal",
    "research publication",
    "citation database",
    "millions of academic papers",
    "academic literature",
    "scientific literature",
    "literature search",
    "paper retrieval",
    "paper discovery",
]

# Compile a single regex for efficiency (word-boundary-aware for short terms)
_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in PAPER_KEYWORDS),
    re.IGNORECASE,
)


def _trim_desc(desc: str) -> str:
    """
    Keep only the 'purpose' part of the description, stripping away
    parameter documentation sections that may contain example queries
    with incidental keyword matches (e.g. "search for AI research papers").
    Splits at the first occurrence of section markers like 'Args:', 'Parameters:',
    'Examples:', 'Returns:', 'Note:'.
    """
    cutoff = re.search(
        # Match section headers whether they start on a new line or inline after a sentence.
        # The "?:" lookbehind isn't needed; we just find the earliest boundary.
        r"(?:(?:\.\s+)|(?:\n\s*))(?:Args|Parameters|Arguments|Examples?|Returns?|Notes?|Raises?)\s*:",
        desc,
        re.IGNORECASE,
    )
    return desc[: cutoff.start()] if cutoff else desc


def is_paper_search_tool(tool: dict) -> bool:
    name = tool["function"]["name"]
    desc = tool["function"].get("description", "")
    # Match the full name but only the purpose section of the description
    text = f"{name} {_trim_desc(desc)}"
    return bool(_PATTERN.search(text))


def main():
    src = Path(__file__).parent / "research_tools_original.json"
    dst = Path(__file__).parent / "research_tools_paper_search.json"

    with src.open(encoding="utf-8") as f:
        tools = json.load(f)

    kept = [t for t in tools if is_paper_search_tool(t)]

    with dst.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    # ---- summary ----
    print(f"Total tools  : {len(tools)}")
    print(f"Kept (paper) : {len(kept)}")
    print(f"Removed      : {len(tools) - len(kept)}")
    print(f"Output       : {dst}")
    print()
    print("Kept tools:")
    for t in kept:
        name = t["function"]["name"]
        desc = t["function"].get("description", "")[:80].replace("\n", " ")
        print(f"  [{t['server_name']}] {name}")
        print(f"      {desc}")


if __name__ == "__main__":
    main()
