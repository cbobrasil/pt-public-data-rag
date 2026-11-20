# pt-public-data-rag

Production-style RAG pipeline over Portuguese public data (INE / dados.gov.pt):
incremental ingestion, chunking, embeddings and vector search on Databricks, with
data governance (PII masking, chunk-level lineage) and retrieval evaluation.

> 🚧 Work in progress. Built step by step as a learning and portfolio project.

## Objective
<!-- What question does this system answer, for whom, and why it is a data engineering problem. -->

## Architecture
<!-- Diagram + short description of each stage: ingestion → parsing → chunking → embeddings → index → serving. -->

## How to run
<!-- Prerequisites, setup, commands. -->

## Evaluation
<!-- Evaluation dataset, metrics (recall@k, ...), results table comparing configurations. -->

## Governance
<!-- PII handling, lineage (document → version → chunk → embedding), access control, cost tracking. -->

## Decisions and trade-offs
<!-- Chunking strategy, vector store choice, index type, incremental refresh strategy, and why. -->

## Repository layout
| Folder | Role |
|---|---|
| `ingestion/` | Download and land raw documents and tables |
| `processing/` | Parsing, cleaning, PII masking, chunking |
| `indexing/` | Embeddings and vector index (incremental refresh) |
| `serving/` | Question-answering endpoint |
| `evaluation/` | Evaluation dataset and metrics |
| `docs/` | Architecture diagrams and notes |
