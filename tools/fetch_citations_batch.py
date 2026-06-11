#!/usr/bin/env python3
"""Batch export citing papers for a fixed list of paper titles."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import asdict
from http.cookiejar import CookieJar
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, build_opener

from scholar_citations import (
    ScholarResult,
    add_start,
    choose_cited_by_url,
    fetch,
    parse_results,
    search_url,
    write_csv,
    write_json,
)


class PartialFetchError(RuntimeError):
    def __init__(self, message: str, rows: list[ScholarResult]) -> None:
        super().__init__(message)
        self.rows = rows


PAPERS = [
    "NeW CRFs: Neural Window Fully-connected CRFs for Monocular Depth Estimation",
    "GIC: Gaussian-Informed Continuum for Physical Property Identification and Simulation",
    "3D Former: Monocular Scene Reconstruction with 3D SDF Transformers",
    "MoGenTS: Motion Generation based on Spatial-Temporal Joint Modeling",
    "LAM: Large Avatar Model for One-shot Animatable Gaussian Head",
    "Freditor: High-Fidelity and Transferable NeRF Editing by Frequency Decomposition",
    "NRRLT: End-to-End Nonprehensile Rearrangement with Deep Reinforcement Learning and Simulation-to-Reality Transfer",
    "MFuseNet: Robust Depth Estimation with Learned Multiscopic Fusion",
    "DRO: Deep Recurrent Optimizer for Structure-from-Motion",
    "OV9D: Open-Vocabulary Category-Level 9D Object Pose and Size Estimation",
    "NRRL: Rearrangement with Nonprehensile Manipulation Using Deep Reinforcement Learning",
    "TPRL: Reinforcement Learning in Topology-based Representation for Human Body Movement with Whole Arm Manipulation",
    "SMMV: Stereo Matching by Self-supervision of Multiscopic Vision",
    "CycleSiam: Self-supervised Object Tracking with Cycle-consistent Siamese Networks",
    "LaMP: Language-Motion Pretraining for Motion Generation, Retrieval, and Captioning",
]


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")[:90]


def find_cited_by(
    title: str,
    base_url: str,
    user_agent: str,
    timeout: int,
    opener,
) -> tuple[str, ScholarResult | None]:
    url = search_url(base_url, f'"{title}"')
    page = fetch(url, user_agent, timeout, opener=opener)
    results, block_reason = parse_results(page, url)
    if block_reason:
        raise RuntimeError(f"search blocked: {block_reason}")
    cited_by_url = choose_cited_by_url(results, title)
    matched = next((row for row in results if row.cited_by_url == cited_by_url), None)
    return cited_by_url, matched


def fetch_all_citations(
    cited_by_url: str,
    user_agent: str,
    timeout: int,
    opener,
    max_pages: int,
    delay: float,
) -> list[ScholarResult]:
    rows: list[ScholarResult] = []
    seen: set[tuple[str, str]] = set()
    for page_index in range(max_pages):
        if page_index and delay > 0:
            time.sleep(delay)
        page_url = add_start(cited_by_url, page_index * 10)
        page = fetch(page_url, user_agent, timeout, opener=opener)
        page_rows, block_reason = parse_results(page, page_url)
        if block_reason:
            raise PartialFetchError(f"citation page blocked: {block_reason}", rows)
        valid_rows = [row for row in page_rows if row.title]
        if not valid_rows:
            break
        for row in valid_rows:
            key = (row.title, row.link)
            if key not in seen:
                rows.append(row)
                seen.add(key)
        if len(valid_rows) < 10:
            break
    return rows


def write_summary(path: Path, summary: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "title",
                "status",
                "count",
                "matched_title",
                "cited_by_text",
                "cited_by_url",
                "json_path",
                "csv_path",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)


def load_existing_summary(path: Path) -> dict[int, dict[str, object]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {int(row["index"]): row for row in csv.DictReader(handle)}


def parse_indices(value: str) -> set[int] | None:
    if not value:
        return None
    indices: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    return indices


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="https://www.gupiaoq.com")
    parser.add_argument("--out-dir", default="tools/citations")
    parser.add_argument("--max-pages", type=int, default=200)
    parser.add_argument("--delay", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--indices", default="", help="Comma-separated 1-based paper indices, e.g. 1,3,15.")
    parser.add_argument(
        "--user-agent",
        default=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opener = build_opener(HTTPCookieProcessor(CookieJar()))
    selected_indices = parse_indices(args.indices)
    summary_by_index = load_existing_summary(out_dir / "summary.csv")

    for index, title in enumerate(PAPERS, start=1):
        if selected_indices is not None and index not in selected_indices:
            continue
        slug = f"{index:02d}_{slugify(title)}"
        json_path = out_dir / f"{slug}.json"
        csv_path = out_dir / f"{slug}.csv"
        cited_by_url = ""
        matched = None
        print(f"[{index}/{len(PAPERS)}] Searching: {title}", flush=True)
        try:
            cited_by_url, matched = find_cited_by(
                title,
                args.base_url,
                args.user_agent,
                args.timeout,
                opener,
            )
            if not cited_by_url:
                raise RuntimeError("no cited-by URL found")
            print(f"  Cited-by: {matched.cited_by_text if matched else cited_by_url}", flush=True)
            rows = fetch_all_citations(
                cited_by_url,
                args.user_agent,
                args.timeout,
                opener,
                args.max_pages,
                args.delay,
            )
            write_json(str(json_path), rows)
            write_csv(str(csv_path), rows)
            summary_by_index[index] = {
                "index": index,
                "title": title,
                "status": "ok",
                "count": len(rows),
                "matched_title": matched.title if matched else "",
                "cited_by_text": matched.cited_by_text if matched else "",
                "cited_by_url": cited_by_url,
                "json_path": str(json_path),
                "csv_path": str(csv_path),
                "error": "",
            }
            print(f"  Saved {len(rows)} rows", flush=True)
        except PartialFetchError as exc:
            rows = exc.rows
            if rows:
                write_json(str(json_path), rows)
                write_csv(str(csv_path), rows)
            summary_by_index[index] = {
                "index": index,
                "title": title,
                "status": "partial",
                "count": len(rows),
                "matched_title": matched.title if "matched" in locals() and matched else "",
                "cited_by_text": matched.cited_by_text if "matched" in locals() and matched else "",
                "cited_by_url": cited_by_url if "cited_by_url" in locals() else "",
                "json_path": str(json_path),
                "csv_path": str(csv_path),
                "error": str(exc),
            }
            print(f"  PARTIAL: saved {len(rows)} rows; {exc}", flush=True)
        except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
            summary_by_index[index] = {
                "index": index,
                "title": title,
                "status": "error",
                "count": 0,
                "matched_title": "",
                "cited_by_text": "",
                "cited_by_url": "",
                "json_path": str(json_path),
                "csv_path": str(csv_path),
                "error": str(exc),
            }
            print(f"  ERROR: {exc}", flush=True)
        summary = [summary_by_index[key] for key in sorted(summary_by_index)]
        write_summary(out_dir / "summary.csv", summary)
        with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        if index < len(PAPERS) and args.delay > 0:
            time.sleep(args.delay)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
