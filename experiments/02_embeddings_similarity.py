"""Experiment 02: what embeddings capture (and what they miss) on the INE corpus.

Questions this answers:
1. What does an embedding look like (dimension, values, norm)?
2. Which press releases end up "close" to each other?
3. Given a question, does nearest-neighbour search find the right release?
4. Are cosine similarity and dot product really the same on normalised vectors?

Runs locally and free with a small open-source model (BAAI/bge-small-en-v1.5, 384 dims).

Usage:
    uv run python experiments/02_embeddings_similarity.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from pypdf import PdfReader

CORPUS = Path("data/raw/ine/destaques/en")
MODEL = "BAAI/bge-small-en-v1.5"

# Hand-picked to include near-duplicates (same topic, different month / flash vs final)
# and different topics within the Prices theme.
WANTED = [
    ("CPI annual rate of change increased to 3.8%", "October 2026"),           # CPI final
    ("CPI annual rate of change estimated to have increased to 3.8%", "October 2026"),  # CPI flash
    ("CPI annual rate of change increased to 3.6%", "September 2026"),         # CPI previous month
    ("Industrial Production Prices increased by 10.2%", "October 2026"),
    ("New housing construction costs rose by 7.4%", "September 2026"),
    ("House prices accelerate in 10 of the 24 most populous municipalities", "2nd Quarter 2026"),
]

QUERIES = [
    "What was the inflation rate in September 2026?",
    "How much did the cost of building new homes go up?",
    "Are energy prices pushing up costs for factories?",
]


def lead_paragraph(pdf: Path) -> str:
    """The opening paragraph of a release: the first line of prose after the ALL-CAPS headers.

    A first version looked for a line starting with "The " and silently returned an empty
    string for releases starting with "In September...". The empty text still produced a
    valid-looking vector, and search quality dropped with no error. Hence the assert.
    """
    text = PdfReader(pdf).pages[0].extract_text()
    start = re.search(r"^[A-Z][a-z].{60,}$", text, flags=re.MULTILINE)  # sentence-case, long line
    lead = text[start.start():] if start else ""
    lead = re.split(r"\n\s*\n|Figure \d", lead)[0]  # stop at the first blank line or chart
    lead = " ".join(lead.split())[:800]
    assert len(lead) > 100, f"lead paragraph not found in {pdf.name}"
    return lead


def short(title: str, period: str) -> str:
    kind = "CPI flash" if "estimated" in title else title.split(" annual")[0].split(" increased")[0]
    return f"{kind[:24]} ({period[:3]}{period[-2:]})"


def main() -> None:
    manifest = [json.loads(line) for line in (CORPUS / "manifest.jsonl").open()]
    docs = []
    for title, period in WANTED:
        row = next(r for r in manifest if r["title"] == title and r["reference_period"] == period)
        docs.append((short(title, period), lead_paragraph(CORPUS / row["file"])))
    labels = [label for label, _ in docs]

    model = TextEmbedding(MODEL)
    # Passages and queries are embedded slightly differently: bge models add an
    # instruction prefix to queries ("Represent this sentence for searching ...").
    doc_vecs = np.array(list(model.passage_embed([text for _, text in docs])))
    query_vecs = np.array(list(model.query_embed(QUERIES)))

    print("\n== 1. What an embedding looks like ==")
    print(f"text:   {docs[0][1][:110]}...")
    print(f"vector: shape={doc_vecs[0].shape}, first values={np.round(doc_vecs[0][:6], 3)}")
    print(f"norm:   {np.linalg.norm(doc_vecs[0]):.4f}  (normalised -> length 1)")

    print("\n== 2. Cosine similarity between releases (1.0 = identical direction) ==")
    sims = doc_vecs @ doc_vecs.T  # dot product == cosine because vectors are normalised
    print(" " * 28 + " ".join(f"{i:>6}" for i in range(len(labels))))
    for i, label in enumerate(labels):
        print(f"{i} {label:26}" + " ".join(f"{sims[i, j]:6.3f}" for j in range(len(labels))))

    print("\n== 3. Search: rank releases by similarity to each question ==")
    for q, qv in zip(QUERIES, query_vecs):
        scores = doc_vecs @ qv
        order = np.argsort(-scores)
        print(f"\nQ: {q}")
        for rank, i in enumerate(order[:3], 1):
            print(f"   {rank}. {scores[i]:.3f}  {labels[i]}")

    print("\n== 4. Cosine vs dot product vs euclidean on normalised vectors ==")
    a, b = doc_vecs[0], doc_vecs[3]
    cosine = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"cosine={cosine:.4f}  dot={a @ b:.4f}  euclidean={np.linalg.norm(a - b):.4f}"
          f"  (euclidean^2 = 2 - 2*cosine = {2 - 2 * cosine:.4f})")


if __name__ == "__main__":
    main()
