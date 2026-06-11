#!/usr/bin/env python3
"""Fill abbreviated citation authors with full names from public metadata APIs."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_FIELDS = [
    "title",
    "authors",
    "authors_original",
    "authors_source",
    "authors_metadata_title",
    "authors_match_score",
    "year",
    "venue",
    "link",
    "snippet",
    "cited_by_text",
    "cited_by_url",
]


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def title_score(left: str, right: str) -> float:
    left_norm = normalize_title(left)
    right_norm = normalize_title(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 1.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


def needs_full_authors(authors: str) -> bool:
    if not authors:
        return True
    if "…" in authors or "..." in authors:
        return True
    for author in authors.split(","):
        tokens = author.strip().split()
        if not tokens:
            continue
        first = tokens[0].replace(".", "")
        if first.isupper() and 1 <= len(first) <= 3:
            return True
    return False


def format_author_name(name: str) -> str:
    parts = [part.strip() for part in name.split(",", 1)]
    if len(parts) == 2 and parts[0] and parts[1]:
        return f"{parts[1]} {parts[0]}"
    return name.strip()


def author_initials(authors: str) -> list[str]:
    initials: list[str] = []
    for author in authors.replace("…", "").replace("...", "").split(","):
        tokens = author.strip().split()
        if not tokens:
            continue
        initials.append(tokens[0][0].upper())
    return initials


def author_initials_match(original_authors: str, full_authors: list[str]) -> bool:
    original = author_initials(original_authors)
    full = author_initials(", ".join(full_authors))
    if not original or not full:
        return False
    comparable = min(len(original), len(full))
    if comparable < 2:
        return False
    return original[:comparable] == full[:comparable]


def request_json(url: str, user_agent: str, timeout: int) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return json.loads(response.read().decode(charset, errors="replace"))


class CitationMetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.authors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "meta":
            return
        values = {key.lower(): value or "" for key, value in attrs}
        name = (values.get("name") or values.get("property") or "").lower()
        content = (values.get("content") or "").strip()
        if not content:
            return
        if name in {"citation_title", "dc.title", "og:title"} and not self.title:
            self.title = content
        elif name in {"citation_author", "dc.creator"}:
            self.authors.append(content)


def page_meta_candidate(title: str, link: str, user_agent: str, timeout: int) -> dict[str, Any] | None:
    if not link.startswith(("http://", "https://")):
        return None
    request = Request(link, headers={"User-Agent": user_agent, "Accept": "text/html"})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "html" not in content_type.lower():
            return None
        charset = response.headers.get_content_charset() or "utf-8"
        page_html = response.read(1_500_000).decode(charset, errors="replace")
    parser = CitationMetaParser()
    parser.feed(page_html)
    metadata_title = parser.title or title
    authors = []
    for author in parser.authors:
        author = format_author_name(author)
        if author and author not in authors:
            authors.append(author)
    if not authors:
        return None
    return {
        "source": "page-meta",
        "metadata_title": metadata_title,
        "authors": authors,
        "score": title_score(title, metadata_title),
    }


def crossref_candidates(title: str, user_agent: str, timeout: int, rows: int) -> list[dict[str, Any]]:
    params = urlencode({"query.title": title, "rows": rows, "select": "title,author"})
    data = request_json(f"https://api.crossref.org/works?{params}", user_agent, timeout)
    items = data.get("message", {}).get("items", [])
    candidates: list[dict[str, Any]] = []
    for item in items:
        titles = item.get("title") or []
        metadata_title = titles[0] if titles else ""
        authors = []
        for author in item.get("author") or []:
            given = (author.get("given") or "").strip()
            family = (author.get("family") or "").strip()
            name = " ".join(part for part in [given, family] if part)
            if name:
                authors.append(name)
        if metadata_title and authors:
            candidates.append(
                {
                    "source": "crossref",
                    "metadata_title": metadata_title,
                    "authors": authors,
                    "score": title_score(title, metadata_title),
                }
            )
    return candidates


def openalex_candidates(title: str, user_agent: str, timeout: int, rows: int) -> list[dict[str, Any]]:
    url = f"https://api.openalex.org/works?{urlencode({'search': title, 'per-page': rows})}"
    data = request_json(url, user_agent, timeout)
    candidates: list[dict[str, Any]] = []
    for item in data.get("results") or []:
        metadata_title = item.get("title") or ""
        authors = []
        for authorship in item.get("authorships") or []:
            display_name = (authorship.get("author") or {}).get("display_name") or ""
            if display_name:
                authors.append(display_name)
        if metadata_title and authors:
            candidates.append(
                {
                    "source": "openalex",
                    "metadata_title": metadata_title,
                    "authors": authors,
                    "score": title_score(title, metadata_title),
                }
            )
    return candidates


def semantic_scholar_candidates(title: str, user_agent: str, timeout: int, rows: int) -> list[dict[str, Any]]:
    params = urlencode({"query": title, "limit": rows, "fields": "title,authors.name"})
    data = request_json(
        f"https://api.semanticscholar.org/graph/v1/paper/search?{params}",
        user_agent,
        timeout,
    )
    candidates: list[dict[str, Any]] = []
    for item in data.get("data") or []:
        metadata_title = item.get("title") or ""
        authors = [(author.get("name") or "").strip() for author in item.get("authors") or []]
        authors = [author for author in authors if author]
        if metadata_title and authors:
            candidates.append(
                {
                    "source": "semanticscholar",
                    "metadata_title": metadata_title,
                    "authors": authors,
                    "score": title_score(title, metadata_title),
                }
            )
    return candidates


def best_author_match(
    title: str,
    user_agent: str,
    timeout: int,
    rows: int,
    min_score: float,
    sources: list[str],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    fetchers = {
        "openalex": openalex_candidates,
        "crossref": crossref_candidates,
        "semanticscholar": semantic_scholar_candidates,
    }
    for source in sources:
        fetch_candidates = fetchers[source]
        try:
            new_candidates = fetch_candidates(title, user_agent, timeout, rows)
            candidates.extend(new_candidates)
            if new_candidates:
                best = max(new_candidates, key=lambda candidate: float(candidate.get("score") or 0.0))
                if float(best.get("score") or 0.0) >= 0.99 and best.get("authors"):
                    return best
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            candidates.append({"source": fetch_candidates.__name__, "error": str(exc), "score": 0.0})
    usable = [candidate for candidate in candidates if candidate.get("authors")]
    if not usable:
        return None
    best = max(usable, key=lambda candidate: float(candidate.get("score") or 0.0))
    if float(best.get("score") or 0.0) < min_score:
        return None
    return best


def load_cache(path: Path) -> dict[str, dict[str, Any] | None]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_cache(path: Path, cache: dict[str, dict[str, Any] | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_rows(json_path: Path, rows: list[dict[str, Any]]) -> None:
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    csv_path = json_path.with_suffix(".csv")
    fields = list(DEFAULT_FIELDS)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def citation_files(citations_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in citations_dir.glob("[0-9][0-9]_*.json")
        if path.name not in {"summary.json"}
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--citations-dir", default="tools/citations")
    parser.add_argument("--cache", default="tools/citations/author_full_name_cache.json")
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.86)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--page-meta", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sources",
        default="openalex,crossref",
        help="Comma-separated metadata sources: openalex,crossref,semanticscholar.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--user-agent",
        default="tac_infra_author_full_name_fill/1.0 (mailto:metadata@example.com)",
    )
    args = parser.parse_args()

    citations_dir = Path(args.citations_dir)
    cache_path = Path(args.cache)
    cache = load_cache(cache_path)
    sources = [source.strip().lower() for source in args.sources.split(",") if source.strip()]
    allowed_sources = {"openalex", "crossref", "semanticscholar"}
    unknown_sources = [source for source in sources if source not in allowed_sources]
    if unknown_sources:
        raise SystemExit(f"unknown sources: {', '.join(unknown_sources)}")
    looked_up = 0
    page_looked_up = 0
    changed = 0
    unresolved = 0

    for json_path in citation_files(citations_dir):
        rows = read_rows(json_path)
        file_changed = False
        for row in rows:
            title = str(row.get("title") or "")
            current_authors = str(row.get("authors") or "")
            if not title or not needs_full_authors(current_authors):
                continue

            cache_key = normalize_title(title)
            if cache_key not in cache:
                if args.limit and looked_up >= args.limit:
                    unresolved += 1
                    continue
                match = best_author_match(
                    title,
                    args.user_agent,
                    args.timeout,
                    args.rows,
                    args.min_score,
                    sources,
                )
                cache[cache_key] = match
                looked_up += 1
                if args.progress_every and looked_up % args.progress_every == 0:
                    print(
                        f"progress looked_up={looked_up} changed={changed} unresolved={unresolved} cache={len(cache)}",
                        flush=True,
                    )
                    if not args.dry_run:
                        save_cache(cache_path, cache)
                if args.delay > 0:
                    time.sleep(args.delay)
            match = cache.get(cache_key)
            if not match:
                link = str(row.get("link") or "")
                if args.page_meta and link:
                    try:
                        page_match = page_meta_candidate(title, link, args.user_agent, args.timeout)
                    except (HTTPError, URLError, TimeoutError, UnicodeError) as exc:
                        page_match = None
                    page_looked_up += 1
                    if (
                        page_match
                        and page_match.get("authors")
                        and (
                            float(page_match.get("score") or 0.0) >= args.min_score
                            or author_initials_match(current_authors, list(page_match["authors"]))
                        )
                    ):
                        match = page_match
                        cache[cache_key] = page_match
                    if args.delay > 0:
                        time.sleep(args.delay)
                if not match:
                    unresolved += 1
                    continue

            full_authors = ", ".join(match["authors"])
            if full_authors and full_authors != current_authors:
                row.setdefault("authors_original", current_authors)
                row["authors"] = full_authors
                row["authors_source"] = match["source"]
                row["authors_metadata_title"] = match["metadata_title"]
                row["authors_match_score"] = f"{float(match['score']):.3f}"
                file_changed = True
                changed += 1

        if file_changed and not args.dry_run:
            write_rows(json_path, rows)
            print(f"updated {json_path}", flush=True)

    if not args.dry_run:
        save_cache(cache_path, cache)
    print(
        f"changed={changed} looked_up={looked_up} page_looked_up={page_looked_up} unresolved={unresolved} cache={len(cache)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
