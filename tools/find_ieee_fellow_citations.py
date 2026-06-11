#!/usr/bin/env python3
"""Find IEEE Fellows among authors of papers citing the target papers."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin
from urllib.request import Request, urlopen


WIKIPEDIA_INDEX = "https://en.wikipedia.org/wiki/Lists_of_fellows_of_the_IEEE"
MANUAL_FELLOW_SOURCES = [
    "https://en.wikipedia.org/wiki/List_of_fellows_of_IEEE_Computer_Society",
    "https://en.wikipedia.org/wiki/List_of_fellows_of_IEEE_Robotics_and_Automation_Society",
    "https://en.wikipedia.org/wiki/List_of_fellows_of_IEEE_Signal_Processing_Society",
    "https://en.wikipedia.org/wiki/List_of_fellows_of_IEEE_Control_Systems_Society",
    "https://en.wikipedia.org/wiki/List_of_fellows_of_IEEE_Computational_Intelligence_Society",
    "https://en.wikipedia.org/wiki/List_of_fellows_of_IEEE_Geoscience_and_Remote_Sensing_Society",
    "https://www.grss-ieee.org/about/membership/fellow-information/ieee-fellows-and-life-fellows/",
]
MANUAL_FELLOWS = {
    "Danica Kragic": "https://www.ieee-ras.org/images/attachments/awards/RAS_Fellow_listing_2025.pdf",
}


@dataclass(frozen=True)
class Fellow:
    name: str
    normalized_name: str
    source: str


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []
            self._in_cell = True
        elif tag == "br" and self._in_cell and self._cell is not None:
            self._cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self._in_cell and self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._cell is not None and self._row is not None:
            self._row.append(display_name("".join(self._cell)))
            self._cell = None
            self._in_cell = False
        elif tag == "tr" and self._row is not None and self._table is not None:
            if any(cell for cell in self._row):
                self._table.append(self._row)
            self._row = None
        elif tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None


def parse_tables(page: str) -> list[list[list[str]]]:
    parser = TableParser()
    parser.feed(page)
    return parser.tables


def fetch_text(url: str, timeout: int = 30) -> str:
    request = Request(url, headers={"User-Agent": "tac-infra-ieee-fellow-match/1.0"})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def normalize_name(value: str) -> str:
    value = html.unescape(str(value))
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"\[[^\]]+\]", " ", value)
    value = re.sub(r"\([^)]*\)", " ", value)
    value = value.replace("\xa0", " ")
    value = value.replace(".", " ")
    value = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ' -]+", " ", value)
    value = re.sub(r"\b(Jr|Sr|II|III|IV)\b", " ", value, flags=re.I)
    return re.sub(r"\s+", " ", value).strip().lower()


def display_name(value: str) -> str:
    value = html.unescape(str(value))
    value = re.sub(r"\[[^\]]+\]", "", value)
    value = re.sub(r"\([^)]*\)", "", value)
    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value).strip(" -\t\r\n")
    return value


def wikipedia_fellow_urls() -> list[str]:
    page = fetch_text(WIKIPEDIA_INDEX)
    urls = set(MANUAL_FELLOW_SOURCES)
    for href in re.findall(r'href="([^"]+)"', page):
        if "/wiki/List_of_fellows_of_IEEE_" in href:
            urls.add(urljoin(WIKIPEDIA_INDEX, href))
    return sorted(urls)


def names_from_table(table: list[list[str]]) -> Iterable[str]:
    if not table:
        return
    columns = [str(column).strip().lower() for column in table[0]]
    for wanted in ("fellow", "name", "first name"):
        for index, column in enumerate(columns):
            if wanted == column or column.endswith(f" {wanted}"):
                if wanted == "first name" and "last name" in columns:
                    last_index = columns.index("last name")
                    middle_index = columns.index("middle initial") if "middle initial" in columns else None
                    for row in table[1:]:
                        if len(row) <= max(index, last_index):
                            continue
                        first = display_name(row[index])
                        middle = display_name(row[middle_index]) if middle_index is not None and len(row) > middle_index else ""
                        last = display_name(row[last_index])
                        yield " ".join(part for part in (first, middle, last) if part)
                else:
                    for row in table[1:]:
                        if len(row) > index:
                            yield display_name(row[index])
                return


def collect_fellows(cache_path: Path, refresh: bool) -> dict[str, Fellow]:
    if cache_path.exists() and not refresh:
        with cache_path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        return {
            row["normalized_name"]: Fellow(row["name"], row["normalized_name"], row["source"])
            for row in data
        }

    fellows: dict[str, Fellow] = {}
    for url in wikipedia_fellow_urls():
        try:
            tables = parse_tables(fetch_text(url))
        except Exception:
            continue
        for table in tables:
            for name in names_from_table(table):
                normalized = normalize_name(name)
                if len(normalized.split()) < 2:
                    continue
                fellows.setdefault(normalized, Fellow(display_name(name), normalized, url))
    for name, source in MANUAL_FELLOWS.items():
        normalized = normalize_name(name)
        fellows.setdefault(normalized, Fellow(name, normalized, source))

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump([fellow.__dict__ for fellow in fellows.values()], handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return fellows


def split_authors(authors: str) -> list[str]:
    parts = [author.strip() for author in authors.split(",") if author.strip()]
    candidates = list(parts)
    for index in range(0, len(parts) - 1, 2):
        if len(parts[index].split()) <= 2 and len(parts[index + 1].split()) <= 3:
            candidates.append(f"{parts[index + 1]} {parts[index]}")
    return candidates


def load_target_titles(summary_csv: Path) -> dict[int, str]:
    with summary_csv.open(encoding="utf-8", newline="") as handle:
        return {int(row["index"]): row["title"] for row in csv.DictReader(handle)}


def citation_index_from_path(path: Path) -> int:
    return int(path.name.split("_", 1)[0])


def find_matches(citations_dir: Path, fellows: dict[str, Fellow]) -> list[dict[str, str]]:
    target_titles = load_target_titles(citations_dir / "summary.csv")
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for csv_path in sorted(citations_dir.glob("[0-9][0-9]_*.csv")):
        cited_index = citation_index_from_path(csv_path)
        cited_title = target_titles[cited_index]
        with csv_path.open(encoding="utf-8", newline="") as handle:
            for citation in csv.DictReader(handle):
                citing_title = citation["title"]
                for author in split_authors(citation.get("authors", "")):
                    normalized = normalize_name(author)
                    fellow = fellows.get(normalized)
                    if not fellow:
                        continue
                    key = (fellow.normalized_name, citing_title, cited_title)
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "fellow_name": fellow.name,
                            "citing_paper": citing_title,
                            "cited_my_paper": cited_title,
                            "citing_paper_year": citation.get("year", ""),
                            "citing_paper_link": citation.get("link", ""),
                            "matched_author_name": author,
                            "fellow_source": fellow.source,
                        }
                    )
    return rows


def write_matches(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "fellow_name",
        "citing_paper",
        "cited_my_paper",
        "citing_paper_year",
        "citing_paper_link",
        "matched_author_name",
        "fellow_source",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--citations-dir", default="tools/citations")
    parser.add_argument("--fellow-cache", default="tools/ieee_fellows_cache.json")
    parser.add_argument("--out", default="tools/ieee_fellow_citation_matches.csv")
    parser.add_argument("--refresh-fellows", action="store_true")
    args = parser.parse_args()

    fellows = collect_fellows(Path(args.fellow_cache), args.refresh_fellows)
    rows = find_matches(Path(args.citations_dir), fellows)
    write_matches(Path(args.out), rows)
    print(f"fellows={len(fellows)} matches={len(rows)} unique_fellows={len({row['fellow_name'] for row in rows})}")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
