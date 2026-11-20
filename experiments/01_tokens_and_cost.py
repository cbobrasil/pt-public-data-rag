"""Experiment 01: tokens, cost and latency on the real corpus.

Questions this answers:
1. How many tokens is an INE press release? Does it depend on the model's tokenizer?
2. Is it cheaper to send the PDF itself or the text we extract from it?
3. What would one LLM pass over the whole corpus cost, per model?
4. (--call) What do latency and real usage look like for one actual request?

Token counting (`count_tokens`) is free. Only `--call` spends money (a fraction of a cent).

Usage:
    uv run python experiments/01_tokens_and_cost.py            # free: counts + estimates
    uv run python experiments/01_tokens_and_cost.py --call     # + one real request to Haiku
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pypdf import PdfReader

CORPUS = Path("data/raw/ine/destaques/en")

# USD per 1M tokens (input, output), Anthropic list prices as of 2026-11.
# Haiku's price applies to prompts up to 100K tokens.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
ASSUMED_OUTPUT_TOKENS = 500  # e.g. a JSON with extracted fields, plus some thinking


def extract_text(pdf: Path) -> tuple[int, str]:
    reader = PdfReader(pdf)
    return len(reader.pages), "\n".join(page.extract_text() or "" for page in reader.pages)


def count_text(client: anthropic.Anthropic, model: str, text: str) -> int:
    return client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": text}]
    ).input_tokens


def count_pdf(client: anthropic.Anthropic, model: str, pdf: Path) -> int:
    """Native PDF input: the API sees the extracted text AND an image of every page."""
    data = base64.standard_b64encode(pdf.read_bytes()).decode()
    doc = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": data}}
    return client.messages.count_tokens(
        model=model, messages=[{"role": "user", "content": [doc]}]
    ).input_tokens


def usd(tokens_in: int, tokens_out: int, model: str) -> float:
    p_in, p_out = PRICES[model]
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--call", action="store_true", help="make one real (paid) request")
    args = parser.parse_args()

    load_dotenv()
    client = anthropic.Anthropic()
    manifest = [json.loads(line) for line in (CORPUS / "manifest.jsonl").open()]
    # One sample per distinct release type, so we don't measure three CPI releases.
    by_type: dict[str, dict] = {}
    for row in manifest:
        by_type.setdefault(row["title"].split()[0], row)
    samples = list(by_type.values())[: args.samples]

    print("\n== 1. Tokens per document (text extracted with pypdf) ==")
    header = f"{'document':45} {'pages':>5} {'chars':>7} " + " ".join(f"{m.split('-')[1]:>8}" for m in PRICES)
    print(header + f" {'chars/tok':>9}")
    totals = {m: 0 for m in PRICES}
    texts = {}
    for row in samples:
        pdf = CORPUS / row["file"]
        pages, text = extract_text(pdf)
        texts[row["file"]] = text
        counts = {m: count_text(client, m, text) for m in PRICES}
        for m in PRICES:
            totals[m] += counts[m]
        ratio = len(text) / counts["claude-opus-5-5"]
        print(f"{row['title'][:45]:45} {pages:5} {len(text):7} "
              + " ".join(f"{counts[m]:8}" for m in PRICES) + f" {ratio:9.1f}")

    print("\n== 2. Native PDF vs extracted text (claude-opus-5-5) ==")
    for row in samples[:2]:
        pdf = CORPUS / row["file"]
        as_text = count_text(client, "claude-opus-5-5", texts[row["file"]])
        as_pdf = count_pdf(client, "claude-opus-5-5", pdf)
        print(f"{row['title'][:45]:45} text={as_text:6}  pdf={as_pdf:6}  ({as_pdf / as_text:.1f}x)")

    print(f"\n== 3. Cost of one pass over the corpus ({len(manifest)} docs, "
          f"{ASSUMED_OUTPUT_TOKENS} output tokens each) ==")
    for m in PRICES:
        avg_in = totals[m] / len(samples)
        total = usd(avg_in, ASSUMED_OUTPUT_TOKENS, m) * len(manifest)
        print(f"{m:20} avg input {avg_in:7.0f} tok/doc -> ${total:.4f} for the corpus")

    if args.call:
        model = "claude-haiku-5-5"
        row = samples[0]
        print(f"\n== 4. One real request ({model}) ==")
        start = time.perf_counter()
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": (
                "In one sentence, what is the headline figure of this press release?\n\n"
                + texts[row["file"]]
            )}],
        )
        latency = time.perf_counter() - start
        answer = next((b.text for b in response.content if b.type == "text"), "")
        u = response.usage
        print(f"answer:   {answer}")
        print(f"latency:  {latency:.2f}s | stop_reason: {response.stop_reason}")
        print(f"usage:    input={u.input_tokens} output={u.output_tokens} (output includes thinking)")
        print(f"cost:     ${usd(u.input_tokens, u.output_tokens, model):.6f} | request_id: {response._request_id}")


if __name__ == "__main__":
    main()
