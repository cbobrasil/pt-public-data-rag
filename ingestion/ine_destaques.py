"""Download INE "Destaques" (press releases) as PDFs into the raw landing zone.

Idempotent and incremental: a JSONL manifest records every downloaded document
(with its SHA-256), and documents already in the manifest are skipped. The
listing is sorted newest first, so re-running only fetches new releases.

Usage:
    python -m ingestion.ine_destaques --theme prices --max-docs 50
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from ingestion.common import fetch, sha256, write_atomic

log = logging.getLogger(__name__)

BASE_URL = "https://www.ine.pt"
LISTING_URL = (
    BASE_URL + "/xportal/xmain?xpid=INE&xpgid=ine_destaques"
    "&DESTAQUEStema={theme_id}&DESTAQUESpagenumber={page}&xlang={lang}"
)
PDF_URL = BASE_URL + "/ngt_server/attachfileu.jsp?look_parentBoui={pdf_id}&att_display=n&att_download=y"

# Theme ids taken from the "Theme" dropdown on the Destaques listing page.
THEMES = {"prices": "5414296", "labour_market": "5414314", "population": "5414321"}

_ITEM_SPLIT = 'class="destaque_item"'
_TITLE_RE = re.compile(r'DESTAQUESdest_boui=(\d+)[^>]*>([^<]+)</a>\s*([^<]*)')
_DATE_RE = re.compile(r'class="datadestaques">([^<]+)')
_PDF_RE = re.compile(r"look_parentBoui=(\d+)")


@dataclass(frozen=True)
class DestaqueRef:
    """One entry of the listing page. `source_id` identifies the release; `pdf_id` its PDF."""

    source_id: str
    pdf_id: str
    title: str
    reference_period: str
    published_date: str  # ISO date


def parse_listing(page_html: str) -> list[DestaqueRef]:
    """Extract releases from one listing page (English version)."""
    refs = []
    for block in page_html.split(_ITEM_SPLIT)[1:]:
        title_m, date_m, pdf_m = _TITLE_RE.search(block), _DATE_RE.search(block), _PDF_RE.search(block)
        if not (title_m and date_m and pdf_m):
            log.warning("Skipping listing item with unexpected structure")
            continue
        period = html.unescape(title_m.group(3)).strip().lstrip("-").strip()
        published = datetime.strptime(date_m.group(1).strip(), "%d %B %Y").date()
        refs.append(
            DestaqueRef(
                source_id=title_m.group(1),
                pdf_id=pdf_m.group(1),
                title=html.unescape(title_m.group(2)).strip(),
                reference_period=period,
                published_date=published.isoformat(),
            )
        )
    return refs


def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {row["source_id"]: row for row in map(json.loads, f) if row}


def run(theme: str, max_docs: int, lang: str, out_dir: Path, max_pages: int = 50) -> int:
    manifest_path = out_dir / "manifest.jsonl"
    manifest = load_manifest(manifest_path)
    log.info("Manifest has %d documents", len(manifest))

    seen, downloaded = 0, 0
    for page in range(1, max_pages + 1):
        url = LISTING_URL.format(theme_id=THEMES[theme], page=page, lang=lang)
        refs = parse_listing(fetch(url).decode("utf-8", errors="replace"))
        if not refs:
            break
        for ref in refs:
            if seen >= max_docs:
                break
            seen += 1
            pdf_path = out_dir / f"{ref.source_id}.pdf"
            if ref.source_id in manifest and pdf_path.exists():
                continue
            pdf_url = PDF_URL.format(pdf_id=ref.pdf_id)
            data = fetch(pdf_url)
            if not data.startswith(b"%PDF"):
                log.error("Not a PDF, skipping: %s (%s)", ref.title, pdf_url)
                continue
            write_atomic(pdf_path, data)
            row = asdict(ref) | {
                "theme": theme,
                "lang": lang,
                "pdf_url": pdf_url,
                "file": pdf_path.name,
                "sha256": sha256(data),
                "size_bytes": len(data),
                "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            # Append only after the file is safely on disk: manifest never points to a missing file.
            with manifest_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            manifest[ref.source_id] = row
            downloaded += 1
            log.info("Downloaded %s | %s (%d KB)", ref.published_date, ref.title, len(data) // 1024)
        if seen >= max_docs:
            break

    log.info("Done: %d new, %d already present", downloaded, seen - downloaded)
    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--theme", choices=sorted(THEMES), default="prices")
    parser.add_argument("--max-docs", type=int, default=50, help="most recent N releases to keep in sync")
    parser.add_argument("--lang", choices=["en", "pt"], default="en")
    parser.add_argument("--out", type=Path, default=Path("data/raw/ine/destaques"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.theme, args.max_docs, args.lang, args.out / args.lang)


if __name__ == "__main__":
    main()
