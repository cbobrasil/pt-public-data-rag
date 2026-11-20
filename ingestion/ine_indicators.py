"""Download INE statistical indicators (JSON API) as raw, immutable snapshots.

Each snapshot is named after the indicator's last-update date reported by INE,
so re-running without a new INE release does not create a new file (idempotent).

API docs: https://www.ine.pt/xportal/xmain?xpid=INE&xpgid=ine_api_db&INST=719281968&xlang=en

Usage:
    python -m ingestion.ine_indicators                 # default CPI indicators
    python -m ingestion.ine_indicators --varcd 0014663
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path

from ingestion.common import fetch, sha256, write_atomic

log = logging.getLogger(__name__)

API_URL = "https://www.ine.pt/ine/json_indicador/pindica.jsp?op=2&varcd={varcd}&Dim1={periods}&lang={lang}"
# Metadata is always requested in EN because period names are parsed ("October 2026").
META_URL = "https://www.ine.pt/ine/json_indicador/pindicaMeta.jsp?varcd={varcd}&lang=EN"

# Codes found in the INE catalog API (xml_indic.jsp?opc=2).
DEFAULT_INDICATORS = {
    "0014663": "CPI year-on-year rate (%) by NUTS II region and COICOP purpose, monthly, base 2025",
    "0014648": "CPI year-on-year rate (%) by special aggregates (e.g. energy, core), monthly, base 2025",
}


class IneApiError(RuntimeError):
    pass


def validate(payload: list[dict]) -> dict:
    """INE returns HTTP 200 even on errors; the real status is in the `Sucesso` field."""
    if not payload or "Verdadeiro" not in payload[0].get("Sucesso", {}):
        raise IneApiError(f"INE API reported failure: {payload[:1]}")
    body = payload[0]
    if not body.get("Dados"):
        raise IneApiError(f"No data for indicator {body.get('IndicadorCod')}")
    return body


def monthly_periods(since: date, until: date) -> list[str]:
    """INE monthly period codes, e.g. 2026-10 -> 'S3A202610'."""
    codes, y, m = [], since.year, since.month
    while (y, m) <= (until.year, until.month):
        codes.append(f"S3A{y}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return codes


def fetch_metadata(varcd: str) -> tuple[date, str]:
    """Return (last available period, last update date) from the fast metadata endpoint."""
    meta = json.loads(fetch(META_URL.format(varcd=varcd), timeout=60, deadline=120))
    meta = meta[0] if isinstance(meta, list) else meta
    last_period = datetime.strptime(meta["UltimoPeriodo"], "%B %Y").date()  # e.g. "October 2026"
    return last_period, meta["DataUltimaAtualizacao"]


def run(varcd: str, lang: str, out_dir: Path, since: date) -> Path:
    """Fetch monthly values of an indicator from `since` up to the last published month.

    Asking for the full history (Dim1=T, back to 1992) is a ~13 MB response that INE
    can take from minutes to the better part of an hour to stream, so we request a window.
    The cheap metadata call runs first: if we already hold a snapshot for INE's latest
    update (the watermark), the heavy download is skipped entirely.
    """
    last_period, last_update = fetch_metadata(varcd)
    path = out_dir / varcd / f"{varcd}_{lang}_since-{since.isoformat()[:7]}_updated-{last_update}.json"
    if path.exists():
        log.info("Up to date (INE last update %s), skipping: %s", last_update, path)
        return path

    periods = monthly_periods(since, last_period)
    url = API_URL.format(varcd=varcd, periods=",".join(periods), lang=lang)
    log.info("Fetching %s: %d months (%s to %s)", varcd, len(periods), periods[0], periods[-1])
    raw = fetch(url, timeout=120, deadline=600)
    validate(json.loads(raw))
    write_atomic(path, raw)
    body = json.loads(raw)[0]
    n_rows = sum(len(v) for v in body["Dados"].values())
    log.info("Saved %s | %d periods, %d rows, sha256=%s", path, len(body["Dados"]), n_rows, sha256(raw)[:12])
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--varcd", action="append", help="indicator code (repeatable)")
    parser.add_argument("--lang", choices=["EN", "PT"], default="EN")
    parser.add_argument("--since", type=date.fromisoformat, default=date(2024, 1, 1), help="YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=Path("data/raw/ine/indicators"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for varcd in args.varcd or DEFAULT_INDICATORS:
        run(varcd, args.lang, args.out, args.since)


if __name__ == "__main__":
    main()
