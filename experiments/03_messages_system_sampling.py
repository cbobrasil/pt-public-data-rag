"""Experiment 03: messages, system prompt and sampling on a real INE press release.

Questions this answers:
1. Does the same request always return the same answer? (non-determinism)
2. What does a system prompt change when the answer is NOT in the document? (grounding)
3. Does the API remember previous turns? (statelessness: history must be resent)
4. Can we still set temperature=0 on current models?

Uses claude-haiku-5-5; the whole run costs well under US$ 0.01.

Usage:
    uv run python experiments/03_messages_system_sampling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pypdf import PdfReader

CORPUS = Path("data/raw/ine/destaques/en")
MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10, 0.50  # USD per 1M tokens

GROUNDED_SYSTEM = (
    "You answer questions about a Statistics Portugal (INE) press release. "
    "Use only facts stated in the press release. If the answer is not in it, reply exactly: "
    "'Not stated in the document.' Answer in one sentence."
)

spent = {"in": 0, "out": 0}


def ask(client: anthropic.Anthropic, messages: list[dict], system: str | None = None) -> str:
    kwargs = {"system": system} if system else {}
    response = client.messages.create(
        model=MODEL, max_tokens=1024, output_config={"effort": "low"}, messages=messages, **kwargs
    )
    spent["in"] += response.usage.input_tokens
    spent["out"] += response.usage.output_tokens
    return next((b.text for b in response.content if b.type == "text"), "").strip()


def main() -> None:
    load_dotenv()
    client = anthropic.Anthropic()
    manifest = [json.loads(line) for line in (CORPUS / "manifest.jsonl").open()]
    row = next(r for r in manifest if r["title"] == "CPI annual rate of change increased to 3.8%")
    doc = PdfReader(CORPUS / row["file"]).pages[0].extract_text()
    print(f"Document: {row['title']} ({row['reference_period']}), page 1 only")

    def with_doc(question: str) -> str:
        return f"<press_release>\n{doc}\n</press_release>\n\n{question}"

    print("\n== 1. Same request, three times ==")
    q = with_doc("Summarise what happened to prices in one short sentence.")
    for i in range(3):
        print(f"  run {i + 1}: {ask(client, [{'role': 'user', 'content': q}])}")

    print("\n== 2. Answer NOT in the document: without vs with a grounding system prompt ==")
    q = with_doc("What was the unemployment rate in Portugal in October 2026?")
    print(f"  no system:   {ask(client, [{'role': 'user', 'content': q}])}")
    print(f"  with system: {ask(client, [{'role': 'user', 'content': q}], system=GROUNDED_SYSTEM)}")

    print("\n== 3. The API has no memory: follow-up question with vs without history ==")
    first_q = with_doc("What was the CPI annual rate in October 2026?")
    first_a = ask(client, [{"role": "user", "content": first_q}], system=GROUNDED_SYSTEM)
    print(f"  turn 1:            {first_a}")
    history = [
        {"role": "user", "content": first_q},
        {"role": "assistant", "content": first_a},
        {"role": "user", "content": "And in the previous month?"},
    ]
    print(f"  turn 2 + history:  {ask(client, history, system=GROUNDED_SYSTEM)}")
    alone = [{"role": "user", "content": "And in the previous month?"}]
    print(f"  turn 2 alone:      {ask(client, alone)}")

    print("\n== 4. temperature=0 on a current model ==")
    try:
        client.messages.create(
            model=MODEL, max_tokens=50, temperature=0,
            messages=[{"role": "user", "content": "Hi"}],
        )
        print("  accepted")
    except TypeError as exc:  # current SDK versions no longer expose the parameter at all
        print(f"  rejected by the SDK before any request was sent: {exc}")
    except anthropic.BadRequestError as exc:  # older SDKs send it and the API refuses (not billed)
        print(f"  rejected by the API (HTTP {exc.status_code}): {exc.message[:160]}")

    cost = (spent["in"] * PRICE_IN + spent["out"] * PRICE_OUT) / 1_000_000
    print(f"\nTotal: {spent['in']} input + {spent['out']} output tokens = US$ {cost:.5f}")


if __name__ == "__main__":
    main()
