import json

import pytest

from ingestion.ine_destaques import load_manifest, parse_listing
from ingestion.ine_indicators import IneApiError, validate

# Trimmed copy of one item from the real INE listing page (English version).
LISTING_ITEM = """
<div class="destaque_item">
  <div class="container">
  <div class="div_dst_lst_title">
    <div class="linkdestaques"><a title="CPI annual rate of change increased to 3.8% - October 2026 (See detail)"
        class="linkdestaques" href="/xportal/xmain?xpid=INE&xpgid=ine_destaques&amp;DESTAQUESdest_boui=770569116&amp;DESTAQUEStema=5414296&amp;DESTAQUESmodo=2">CPI annual rate of change increased to 3.8%</a>
         - October 2026
    </div><br>
    <div class="datadestaques">11 November 2026<br><br></div>
  </div>
  <div class="div_dir_files">
    <a title="Download the document" href="/ngt_server/attachfileu.jsp?look_parentBoui=828136069&amp;att_display=n&amp;att_download=y"><img src="/img/ficheiros/PDF.gif"></a>
  </div>
  </div>
</div>
"""


def test_parse_listing_extracts_fields():
    [ref] = parse_listing("<html>" + LISTING_ITEM + "</html>")
    assert ref.source_id == "770569116"
    assert ref.pdf_id == "828136069"  # differs from source_id: must be scraped, not derived
    assert ref.title == "CPI annual rate of change increased to 3.8%"
    assert ref.reference_period == "October 2026"
    assert ref.published_date == "2026-11-11"


def test_parse_listing_skips_malformed_items():
    broken = '<div class="destaque_item"><p>layout changed</p></div>'
    assert parse_listing(broken + LISTING_ITEM) and len(parse_listing(broken + LISTING_ITEM)) == 1


def test_parse_listing_empty_page():
    assert parse_listing("<html>no results</html>") == []


def test_load_manifest(tmp_path):
    path = tmp_path / "manifest.jsonl"
    assert load_manifest(path) == {}
    path.write_text(json.dumps({"source_id": "1", "sha256": "abc"}) + "\n")
    assert load_manifest(path) == {"1": {"source_id": "1", "sha256": "abc"}}


def test_validate_rejects_api_failure():
    with pytest.raises(IneApiError):
        validate([{"Sucesso": {"Falso": [{"Msg": "Invalid code"}]}}])


def test_validate_rejects_empty_data():
    with pytest.raises(IneApiError):
        validate([{"Sucesso": {"Verdadeiro": [{"Msg": "OK"}]}, "IndicadorCod": "x", "Dados": {}}])


def test_monthly_periods_crosses_year_boundary():
    from datetime import date

    from ingestion.ine_indicators import monthly_periods

    assert monthly_periods(date(2025, 11, 1), date(2026, 2, 15)) == ["S3A202511", "S3A202512", "S3A202601", "S3A202602"]


def test_read_with_deadline_aborts_slow_stream(monkeypatch):
    import io

    from ingestion import common

    clock = iter(range(0, 1000, 10))  # each call to monotonic() advances 10 s
    monkeypatch.setattr(common.time, "monotonic", lambda: next(clock))
    with pytest.raises(common.DeadlineExceeded):
        common._read_with_deadline(io.BytesIO(b"x" * 1_000_000), deadline=25)
