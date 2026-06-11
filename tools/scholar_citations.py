#!/usr/bin/env python3
"""Fetch citing papers from a Google Scholar-compatible mirror.

This script only reads public result pages. It does not solve CAPTCHAs, log in,
or bypass rate limits.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
from http.cookiejar import CookieJar
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, urlencode, urljoin, urlparse, urlunparse
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen


DEFAULT_TITLE = (
    "GIC: Gaussian-Informed Continuum for Physical Property Identification "
    "and Simulation"
)


@dataclass
class ScholarResult:
    title: str = ""
    authors: str = ""
    year: str = ""
    venue: str = ""
    link: str = ""
    snippet: str = ""
    cited_by_text: str = ""
    cited_by_url: str = ""


def classes(attrs: list[tuple[str, str | None]]) -> set[str]:
    for key, value in attrs:
        if key == "class" and value:
            return set(value.split())
    return set()


def attr(attrs: list[tuple[str, str | None]], name: str) -> str:
    for key, value in attrs:
        if key == name and value:
            return value
    return ""


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


class ScholarParser(HTMLParser):
    """Small parser for Google Scholar's common result-card markup."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.results: list[ScholarResult] = []
        self._current: ScholarResult | None = None
        self._stack: list[set[str]] = []
        self._capture: str | None = None
        self._buffer: list[str] = []
        self._pending_href = ""
        self.block_reason = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        cls = classes(attrs)
        self._stack.append(cls)

        if tag == "form" and "gs_captcha_f" in cls:
            self.block_reason = "captcha"

        if tag == "div" and {"gs_r", "gs_or"} <= cls:
            if self._current:
                self.results.append(self._current)
            self._current = ScholarResult()

        if not self._current:
            return

        if tag == "h3" and "gs_rt" in cls:
            self._start_capture("title")
            return

        if tag == "div" and "gs_a" in cls:
            self._start_capture("authors")
            return

        if tag == "div" and "gs_rs" in cls:
            self._start_capture("snippet")
            return

        if tag == "a":
            href = attr(attrs, "href")
            if self._capture == "title" and not self._current.link and href:
                self._current.link = urljoin(self.base_url, href)
                return
            if self._capture is None and self._in_footer() and href:
                self._pending_href = urljoin(self.base_url, href)
                self._start_capture("footer_link")

    def handle_data(self, data: str) -> None:
        lower = data.lower()
        if "unusual traffic" in lower or "not a robot" in lower:
            self.block_reason = "anti-bot page"
        if "安全验证" in data or "请点击图片中的" in data:
            self.block_reason = "site security verification"
        if self._capture:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._capture and (
            (self._capture in {"title"} and tag == "h3")
            or (self._capture in {"authors", "snippet"} and tag == "div")
            or (self._capture == "footer_link" and tag == "a")
        ):
            text = clean_text("".join(self._buffer))
            if self._current:
                if self._capture == "title":
                    self._current.title = re.sub(r"^\[[A-Z]+\]\s*", "", text)
                elif self._capture == "authors":
                    self._set_authors_metadata(text)
                elif self._capture == "snippet":
                    self._current.snippet = text
                elif self._capture == "footer_link" and _looks_like_cited_by(text):
                    self._current.cited_by_text = text
                    self._current.cited_by_url = self._pending_href
            self._capture = None
            self._buffer = []
            self._pending_href = ""

        if tag == "div" and self._current and self._stack:
            cls = self._stack[-1]
            if {"gs_r", "gs_or"} <= cls:
                self.results.append(self._current)
                self._current = None

        if self._stack:
            self._stack.pop()

    def close(self) -> None:
        super().close()
        if self._current:
            self.results.append(self._current)
            self._current = None

    def _start_capture(self, name: str) -> None:
        self._capture = name
        self._buffer = []

    def _in_footer(self) -> bool:
        return any("gs_fl" in cls for cls in self._stack)

    def _set_authors_metadata(self, text: str) -> None:
        if not self._current:
            return
        parts = [part.strip() for part in text.split(" - ")]
        self._current.authors = parts[0] if parts else text
        self._current.venue = parts[1] if len(parts) > 1 else ""
        year_match = re.search(r"\b(19|20)\d{2}\b", text)
        self._current.year = year_match.group(0) if year_match else ""


def _looks_like_cited_by(text: str) -> bool:
    normalized = text.lower()
    return (
        "cited by" in normalized
        or "被引用" in normalized
        or "引用" in normalized and re.search(r"\d", normalized) is not None
    )


def fetch(
    url: str,
    user_agent: str,
    timeout: int,
    cookie: str = "",
    opener=None,
    retry_cookie_redirect: bool = True,
) -> str:
    headers = {"User-Agent": user_agent}
    if cookie:
        headers["Cookie"] = cookie
    request = Request(url, headers=headers)
    open_url = opener.open if opener else urlopen
    try:
        with open_url(request, timeout=timeout) as response:
            content_type = response.headers.get_content_charset() or "utf-8"
            return response.read().decode(content_type, errors="replace")
    except HTTPError as exc:
        content_type = exc.headers.get_content_charset() or "utf-8"
        body = exc.read().decode(content_type, errors="replace")
        if (
            exc.code == 403
            and retry_cookie_redirect
            and "window.location.href" in body
            and opener is not None
        ):
            time.sleep(1)
            return fetch(
                url,
                user_agent,
                timeout,
                cookie=cookie,
                opener=opener,
                retry_cookie_redirect=False,
            )
        raise HTTPError(exc.url, exc.code, body or exc.reason, exc.headers, None)


def parse_results(page_html: str, base_url: str) -> tuple[list[ScholarResult], str]:
    parser = ScholarParser(base_url)
    parser.feed(page_html)
    parser.close()
    return parser.results, parser.block_reason


def search_url(base_url: str, query: str, start: int = 0) -> str:
    base = base_url.rstrip("/") + "/scholar"
    params = {
        "hl": "en",
        "as_sdt": "0,5",
        "q": query,
    }
    if start:
        params["start"] = str(start)
    return base + "?" + urlencode(params)


def add_start(url: str, start: int) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query["start"] = [str(start)]
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


def choose_cited_by_url(results: Iterable[ScholarResult], title: str) -> str:
    normalized_title = normalize_title(title)
    fallback = ""
    for result in results:
        if not result.cited_by_url:
            continue
        if not fallback:
            fallback = result.cited_by_url
        if normalize_title(result.title) == normalized_title:
            return result.cited_by_url
    return fallback


def normalize_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def write_json(path: str, rows: list[ScholarResult]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump([asdict(row) for row in rows], handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_csv(path: str, rows: list[ScholarResult]) -> None:
    fields = list(ScholarResult.__dataclass_fields__.keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--base-url", default="https://sc.panda985.com")
    parser.add_argument("--cites-url", default="", help="Skip title search and use this cited-by URL.")
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--delay", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--cookie", default="", help="Cookie header from a manually verified browser session.")
    parser.add_argument("--json", default="citing_papers.json")
    parser.add_argument("--csv", default="citing_papers.csv")
    parser.add_argument("--dump-html-on-fail", default="", help="Write the last fetched HTML here on parse failure.")
    parser.add_argument(
        "--user-agent",
        default=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        ),
    )
    args = parser.parse_args()
    opener = build_opener(HTTPCookieProcessor(CookieJar()))

    cited_by_url = args.cites_url
    if not cited_by_url:
        url = search_url(args.base_url, f'"{args.title}"')
        print(f"Searching: {url}", file=sys.stderr)
        try:
            page = fetch(url, args.user_agent, args.timeout, args.cookie, opener=opener)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"Search request failed: {exc}", file=sys.stderr)
            return 2
        results, block_reason = parse_results(page, url)
        if block_reason:
            print(f"Search page appears blocked: {block_reason}", file=sys.stderr)
            return 3
        cited_by_url = choose_cited_by_url(results, args.title)
        if not cited_by_url:
            if args.dump_html_on_fail:
                with open(args.dump_html_on_fail, "w", encoding="utf-8") as handle:
                    handle.write(page)
            print("No cited-by link found on the search result page.", file=sys.stderr)
            print("Try opening the mirror manually and pass --cites-url.", file=sys.stderr)
            return 4
        print(f"Cited-by URL: {cited_by_url}", file=sys.stderr)

    all_rows: list[ScholarResult] = []
    seen: set[tuple[str, str]] = set()
    for page_index in range(args.max_pages):
        if page_index and args.delay > 0:
            time.sleep(args.delay)
        page_url = add_start(cited_by_url, page_index * 10)
        print(f"Fetching citations page {page_index + 1}: {page_url}", file=sys.stderr)
        try:
            page = fetch(page_url, args.user_agent, args.timeout, args.cookie, opener=opener)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"Citation request failed: {exc}", file=sys.stderr)
            return 2
        rows, block_reason = parse_results(page, page_url)
        if block_reason:
            print(f"Citation page appears blocked: {block_reason}", file=sys.stderr)
            return 3
        if not rows:
            if args.dump_html_on_fail:
                with open(args.dump_html_on_fail, "w", encoding="utf-8") as handle:
                    handle.write(page)
            break
        for row in rows:
            key = (row.title, row.link)
            if row.title and key not in seen:
                all_rows.append(row)
                seen.add(key)

    write_json(args.json, all_rows)
    write_csv(args.csv, all_rows)
    print(f"Saved {len(all_rows)} citing papers to {args.json} and {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
