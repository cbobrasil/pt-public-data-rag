"""Experiment 04: structured output and tool calling on the INE corpus.

Part A - structured output: extract a fixed set of fields from every CPI press release
and check the extracted headline rate against INE's official CPI table (ground truth).
Part B - tool calling: give the model a tool backed by the regional CPI table and watch
the request -> tool_use -> tool_result -> answer loop step by step.

Uses claude-haiku-5-5; the whole run costs about one US cent.

Usage:
    uv run python experiments/04_structured_output_tools.py
"""

from __future__ import annotations

import glob
import json
from datetime import datetime
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pypdf import PdfReader

CORPUS = Path("data/raw/ine/destaques/en")
MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10, 0.50  # USD per 1M tokens
spent = {"in": 0, "out": 0}


# ---------------------------------------------------------------- Part A: structured output

class ReleaseFacts(BaseModel):
    """The fields we want from each press release. Descriptions are sent to the model."""

    indicator: str = Field(description="Name of the statistic, e.g. 'Consumer Price Index'")
    reference_period: str = Field(description="Period the figures refer to, as YYYY-MM")
    is_flash_estimate: bool = Field(description="True if the release is a flash (preliminary) estimate")
    headline_rate_pct: float | None = Field(description="Main year-on-year rate of change, in percent")
    previous_month_rate_pct: float | None = Field(description="Same rate for the previous month, in percent")
    main_driver: str | None = Field(description="Main driver of the change, as a short phrase, if stated")


SYSTEM = (
    "You extract facts from Statistics Portugal (INE) press releases. "
    "Use only what is stated in the text. Use null for anything not stated."
)


def extract(client: anthropic.Anthropic, text: str) -> ReleaseFacts:
    response = client.messages.parse(
        model=MODEL,
        max_tokens=2048,
        output_config={"effort": "low"},
        system=SYSTEM,
        messages=[{"role": "user", "content": f"<press_release>\n{text}\n</press_release>"}],
        output_format=ReleaseFacts,
    )
    spent["in"] += response.usage.input_tokens
    spent["out"] += response.usage.output_tokens
    return response.parsed_output  # already a validated ReleaseFacts object


def official_cpi() -> dict[str, float]:
    """Ground truth: Portugal, all-items CPI year-on-year rate, keyed by 'October 2026'."""
    path = sorted(glob.glob("data/raw/ine/indicators/0014648/*.json"))[-1]
    data = json.load(open(path))[0]["Dados"]
    return {
        period: float(r["valor"])
        for period, rows in data.items()
        for r in rows
        if r["geocod"] == "PT" and r["dim_3_t"] == "Total"
    }


def part_a(client: anthropic.Anthropic) -> None:
    print("== Part A: structured extraction of every CPI release, checked against the official table ==")
    truth = official_cpi()
    manifest = [json.loads(line) for line in (CORPUS / "manifest.jsonl").open()]
    cpi_rows = [r for r in manifest if r["title"].startswith("CPI")]

    first = True
    hits = total = 0
    print(f"{'period':15} {'type':6} {'extracted':>9} {'official':>8}  check")
    for row in sorted(cpi_rows, key=lambda r: r["published_date"]):
        text = PdfReader(CORPUS / row["file"]).pages[0].extract_text()
        facts = extract(client, text)
        if first:
            print(f"(example of one full extraction)\n{facts.model_dump_json(indent=2)}\n")
            first = False
        official = truth.get(row["reference_period"])
        kind = "flash" if facts.is_flash_estimate else "final"
        # Press releases round to 1 decimal while the table keeps 2 (1.9 vs 1.92), so an exact
        # comparison flags rounding as an error. Compare within the release's precision instead.
        ok = (official is not None and facts.headline_rate_pct is not None
              and abs(facts.headline_rate_pct - official) <= 0.05 + 1e-9)
        hits += ok
        total += official is not None
        note = "OK" if ok else ("no official value" if official is None else "DIFFERENT")
        print(f"{row['reference_period']:15} {kind:6} {facts.headline_rate_pct!s:>9} {official!s:>8}  {note}")
    print(f"\nMatches: {hits}/{total}")


# ---------------------------------------------------------------- Part B: tool calling

CPI_TOOL = {
    "name": "get_cpi_rate",
    "description": (
        "Official year-on-year CPI rate (%) from Statistics Portugal for one region and month. "
        "Regions: Portugal, Continente, Norte, Centro, Grande Lisboa, Península de Setúbal, "
        "Alentejo, Algarve, Oeste e Vale do Tejo, Região Autónoma dos Açores, Região Autónoma da Madeira."
    ),
    "strict": True,  # the API guarantees the call's input matches this schema
    "input_schema": {
        "type": "object",
        "properties": {
            "region": {"type": "string", "description": "Region name exactly as listed"},
            "month": {"type": "string", "description": "Month as YYYY-MM"},
        },
        "required": ["region", "month"],
        "additionalProperties": False,
    },
}


def get_cpi_rate(region: str, month: str) -> str:
    """The tool implementation: plain Python reading the regional CPI table. The model never runs this."""
    path = sorted(glob.glob("data/raw/ine/indicators/0014663/*.json"))[-1]
    data = json.load(open(path))[0]["Dados"]
    period = datetime.strptime(month, "%Y-%m").strftime("%B %Y")  # "2026-08" -> "August 2026"
    for r in data.get(period, []):
        if r["geodsg"] == region and r["dim_3_t"] == "Total":
            return json.dumps({"region": region, "month": month, "cpi_yoy_pct": float(r["valor"])})
    return json.dumps({"error": f"no data for region={region!r} month={month!r}"})


def part_b(client: anthropic.Anthropic) -> None:
    print("\n== Part B: tool calling, step by step ==")
    question = "In August 2026, was inflation higher in the Norte region or in the Algarve? By how much?"
    messages = [{"role": "user", "content": question}]
    print(f"USER: {question}")

    for step in range(1, 6):  # safety cap on round trips
        response = client.messages.create(
            model=MODEL, max_tokens=2048, output_config={"effort": "low"},
            tools=[CPI_TOOL], messages=messages,
        )
        spent["in"] += response.usage.input_tokens
        spent["out"] += response.usage.output_tokens
        print(f"\n[round trip {step}] stop_reason={response.stop_reason}")

        if response.stop_reason != "tool_use":
            print("CLAUDE:", next((b.text for b in response.content if b.type == "text"), ""))
            return

        # The model asked for one or more tool calls: run them and send ALL results back together.
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = get_cpi_rate(**block.input)
                print(f"  CLAUDE asks: {block.name}({block.input})")
                print(f"  OUR CODE returns: {output}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


def main() -> None:
    load_dotenv()
    client = anthropic.Anthropic()
    part_a(client)
    part_b(client)
    cost = (spent["in"] * PRICE_IN + spent["out"] * PRICE_OUT) / 1_000_000
    print(f"\nTotal: {spent['in']} input + {spent['out']} output tokens = US$ {cost:.5f}")


if __name__ == "__main__":
    main()
