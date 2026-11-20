"""Experiment 05: real-time calls vs the Message Batches API, plus retries and rate limits.

Questions this answers:
1. What does the API tell us about our rate limits, and what does the SDK do when a call fails?
2. Same 19 CPI extractions as experiment 04: how long and how much, one call at a time?
3. The same 19 requests as ONE batch: how long until results, how much, are the answers the same?

Uses claude-haiku-5-5; the whole run costs well under one US cent.
A batch usually finishes in minutes but may take up to 24 h. If the script is interrupted,
resume polling instead of submitting (and paying for) the batch again:

Usage:
    uv run python experiments/05_batch_vs_realtime.py
    uv run python experiments/05_batch_vs_realtime.py --resume msgbatch_...
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pypdf import PdfReader

exp04 = importlib.import_module("04_structured_output_tools")  # reuse schema + prompt
ReleaseFacts, SYSTEM = exp04.ReleaseFacts, exp04.SYSTEM

CORPUS = Path("data/raw/ine/destaques/en")
MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.10, 0.50  # USD per 1M tokens, real-time; the Batches API charges half
SCHEMA = anthropic.transform_schema(ReleaseFacts)  # what messages.parse() builds under the hood


def request_params(text: str) -> dict:
    """One extraction request. Identical for real-time and batch, so the comparison is fair."""
    return {
        "model": MODEL,
        "max_tokens": 2048,
        "system": SYSTEM,
        "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        "messages": [{"role": "user", "content": f"<press_release>\n{text}\n</press_release>"}],
    }


def facts_from(message) -> ReleaseFacts:
    text = next(b.text for b in message.content if b.type == "text")
    return ReleaseFacts.model_validate_json(text)


def usd(tokens_in: int, tokens_out: int, discount: float = 1.0) -> float:
    return (tokens_in * PRICE_IN + tokens_out * PRICE_OUT) / 1_000_000 * discount


def part_1_retries() -> None:
    print("== 1. What the SDK does when a request fails ==")
    # Point a client at a port where nothing listens: every attempt fails with a connection error.
    # The SDK retries connection errors, 408, 409, 429 and 5xx with exponential backoff + jitter.
    logging.basicConfig(format="  [sdk log] %(message)s")
    logging.getLogger("anthropic").setLevel(logging.INFO)
    broken = anthropic.Anthropic(base_url="http://127.0.0.1:9", max_retries=3, api_key="unused")
    start = time.perf_counter()
    try:
        broken.messages.create(model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "Hi"}])
    except anthropic.APIConnectionError as exc:
        print(f"  gave up after 1 try + 3 retries, {time.perf_counter() - start:.1f}s: {type(exc).__name__}")
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def part_2_realtime(client: anthropic.Anthropic, docs: list[dict]) -> dict[str, ReleaseFacts]:
    print("\n== 2. Real-time: one request per document, one after the other ==")
    results, tok_in, tok_out, latencies = {}, 0, 0, []
    start = time.perf_counter()
    for i, doc in enumerate(docs):
        t0 = time.perf_counter()
        # with_raw_response exposes the HTTP headers, where the rate-limit counters live
        raw = client.messages.with_raw_response.create(**request_params(doc["text"]))
        message = raw.parse()
        latencies.append(time.perf_counter() - t0)
        if i == 0:
            print("  rate-limit headers on the first response:")
            for name, value in sorted(raw.headers.items()):
                if name.startswith("anthropic-ratelimit"):
                    print(f"    {name}: {value}")
        results[doc["source_id"]] = facts_from(message)
        tok_in += message.usage.input_tokens
        tok_out += message.usage.output_tokens
    wall = time.perf_counter() - start
    print(f"  {len(docs)} requests | wall clock {wall:.1f}s | per request: "
          f"min {min(latencies):.1f}s, max {max(latencies):.1f}s")
    print(f"  tokens: {tok_in} in + {tok_out} out = US$ {usd(tok_in, tok_out):.5f}")
    return results


def part_3_batch(client: anthropic.Anthropic, docs: list[dict], resume: str | None) -> dict[str, ReleaseFacts]:
    print("\n== 3. Batch: the same requests submitted as one job ==")
    start = time.perf_counter()
    if resume:
        batch = client.messages.batches.retrieve(resume)
        print(f"  resuming {batch.id}")
    else:
        batch = client.messages.batches.create(requests=[
            # custom_id is how each result finds its way back to the input row: results are NOT
            # guaranteed to come back in submission order. Use a stable business key.
            {"custom_id": doc["source_id"], "params": request_params(doc["text"])} for doc in docs
        ])
        print(f"  submitted {batch.id} (save this id to resume with --resume)")

    while batch.processing_status != "ended":
        c = batch.request_counts
        print(f"  t={time.perf_counter() - start:5.0f}s status={batch.processing_status} "
              f"processing={c.processing} succeeded={c.succeeded} errored={c.errored}")
        time.sleep(15)
        batch = client.messages.batches.retrieve(batch.id)
    print(f"  ended after {time.perf_counter() - start:.0f}s (from this script's point of view)")

    results, tok_in, tok_out, failed = {}, 0, 0, []
    for item in client.messages.batches.results(batch.id):
        if item.result.type == "succeeded":
            message = item.result.message
            results[item.custom_id] = facts_from(message)
            tok_in += message.usage.input_tokens
            tok_out += message.usage.output_tokens
        else:  # errored / canceled / expired: in a pipeline these rows go to a retry queue
            failed.append((item.custom_id, item.result.type))
    print(f"  succeeded {len(results)}, failed {failed or 0}")
    print(f"  tokens: {tok_in} in + {tok_out} out = US$ {usd(tok_in, tok_out, 0.5):.5f} "
          f"(same tokens in real time: US$ {usd(tok_in, tok_out):.5f})")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="batch id to resume polling instead of submitting again")
    args = parser.parse_args()

    load_dotenv()
    client = anthropic.Anthropic()
    manifest = [json.loads(line) for line in (CORPUS / "manifest.jsonl").open()]
    docs = [
        {"source_id": r["source_id"], "period": r["reference_period"],
         "text": PdfReader(CORPUS / r["file"]).pages[0].extract_text()}
        for r in sorted(manifest, key=lambda r: r["published_date"]) if r["title"].startswith("CPI")
    ]

    part_1_retries()
    realtime = part_2_realtime(client, docs)
    batch = part_3_batch(client, docs, args.resume)

    print("\n== 4. Do both modes give the same answers? ==")
    same = sum(realtime[d["source_id"]] == batch.get(d["source_id"]) for d in docs)
    print(f"  identical extractions: {same}/{len(docs)}")
    for d in docs:
        a, b = realtime[d["source_id"]], batch.get(d["source_id"])
        if a != b:
            diff = {k: (v, getattr(b, k, None)) for k, v in a.model_dump().items() if getattr(b, k, None) != v}
            print(f"  {d['period']:15} real-time vs batch: {diff}")


if __name__ == "__main__":
    main()
