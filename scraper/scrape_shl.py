"""
Scrapes the SHL product catalog, keeping only "Individual Test Solutions"
(excludes "Pre-packaged Job Solutions" per the assignment scope).

Run this LOCALLY (not in Claude's sandbox — shl.com isn't on the allowed
network list there). Output: data/catalog.json

Usage:
    pip install requests beautifulsoup4
    python scrape_shl.py --out ../data/catalog.json

Notes / things to verify once you actually hit the live page (I can't
render it from here, so treat the selectors below as a strong starting
point, not gospel):
  - The catalog page renders two tables/sections, one per solution type.
    We locate the "Individual Test Solutions" heading and only parse the
    table that follows it.
  - Pagination is typically via a `start=` or `page=` query param with a
    "Next" link — we follow it until it disappears.
  - Each row typically has: name+link, Remote Testing (dot icon =yes),
    Adaptive/IRT (dot icon = yes), and single-letter Test Type badges
    (A, B, C, D, E, K, P, S).
  - If SHL's markup differs from what's coded here, open the page in a
    browser, view source around one catalog row, and adjust the CSS
    selectors marked with `# ADJUST ME` below.
"""
import argparse
import json
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; SHL-catalog-scraper/1.0)"}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def fetch(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def find_individual_section(soup: BeautifulSoup):
    """Locate the table/section for 'Individual Test Solutions'."""
    # ADJUST ME: heading text match is the most robust anchor across
    # markup changes; fall back to scanning all tables if not found.
    heading = soup.find(
        lambda tag: tag.name in ("h2", "h3", "h4")
        and "individual test solutions" in tag.get_text(strip=True).lower()
    )
    if heading:
        table = heading.find_next("table")
        if table:
            return table
    tables = soup.find_all("table")
    return tables[-1] if tables else None


def parse_row(row) -> dict | None:
    cells = row.find_all("td")
    if not cells:
        return None
    link_tag = row.find("a", href=True)
    if not link_tag:
        return None
    name = link_tag.get_text(strip=True)
    url = urljoin(BASE, link_tag["href"])

    row_text = row.get_text(" ", strip=True)
    remote = _has_marker(row, "remote")
    adaptive = _has_marker(row, "adaptive") or _has_marker(row, "irt")

    test_types = []
    for cell in cells:
        for span in cell.find_all(["span", "a"]):
            t = span.get_text(strip=True)
            if t in TEST_TYPE_MAP:
                test_types.append(t)
    test_type = "".join(sorted(set(test_types))) or "Unknown"

    return {
        "name": name,
        "url": url,
        "test_type": test_type,
        "description": "",
        "duration_minutes": None,
        "remote_testing": remote,
        "adaptive_irt": adaptive,
        "job_levels": [],
        "languages": [],
        "_raw_row_text": row_text,
    }


def _has_marker(row, keyword: str) -> bool | None:
    # ADJUST ME: SHL typically marks yes/no with a filled/outline dot icon
    # (svg or span class like "catalogue__circle -yes"). Adjust the class
    # match once you inspect real markup.
    el = row.find(attrs={"class": lambda c: c and keyword in " ".join(c).lower()})
    if el is None:
        return None
    classes = " ".join(el.get("class", [])).lower()
    return "yes" in classes or "-yes" in classes or "true" in classes


def get_next_page_url(soup: BeautifulSoup, current_url: str) -> str | None:
    # ADJUST ME: look for rel="next" or a link/button with text "Next"
    next_link = soup.find("a", rel="next") or soup.find(
        "a", string=lambda s: s and s.strip().lower() == "next"
    )
    if next_link and next_link.get("href"):
        return urljoin(current_url, next_link["href"])
    return None


def scrape_all(start_url: str = CATALOG_URL, delay: float = 1.0) -> list[dict]:
    items, seen_urls = [], set()
    url = start_url
    page = 0
    while url:
        page += 1
        print(f"[scrape] page {page}: {url}")
        soup = fetch(url)
        table = find_individual_section(soup)
        if table:
            for row in table.find_all("tr"):
                item = parse_row(row)
                if item and item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    items.append(item)
        url = get_next_page_url(soup, url)
        if url:
            time.sleep(delay)
    return items


def enrich_descriptions(items: list[dict], delay: float = 1.0) -> None:
    """Optional second pass: visit each assessment page for a description.
    Slower (N requests) — skip with --no-enrich if you just need names/urls/types."""
    for item in items:
        try:
            soup = fetch(item["url"])
            # ADJUST ME: description is usually the first <p> in the main
            # content area, or a meta description tag.
            meta = soup.find("meta", attrs={"name": "description"})
            if meta and meta.get("content"):
                item["description"] = meta["content"].strip()
            else:
                p = soup.find("p")
                item["description"] = p.get_text(strip=True) if p else ""
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to enrich {item['url']}: {e}")
        time.sleep(delay)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/catalog.json")
    ap.add_argument("--no-enrich", action="store_true", help="skip per-item description fetch")
    args = ap.parse_args()

    items = scrape_all()
    if not args.no_enrich:
        enrich_descriptions(items)

    for item in items:
        item.pop("_raw_row_text", None)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    print(f"[done] wrote {len(items)} items -> {args.out}")


if __name__ == "__main__":
    main()
