# Contract QA — Retrieval-Augmented Question Answering over Legal Contracts

Question answering over commercial contracts, built so that every answer can be traced
back to the clause it came from. Retrieval quality is measured against lawyer-written
annotations rather than judged by eye.

**Corpus** — [CUAD](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus
Dataset): 510 commercial contracts, 13,823 clause spans annotated by lawyers across 41
clause types (Governing Law, Exclusivity, Cap On Liability, Anti-Assignment, …).

**Stack** — Python · PostgreSQL + pgvector · `BAAI/bge-small-en-v1.5` (384-dim, runs on CPU)
· Docker Compose

---

## Why retrieval is necessary here

The median contract in the corpus is 33,143 characters (~8k tokens); the largest is
338,211 (~85k tokens). Sending one contract per query is already expensive, and the full
corpus is impossible. Answers therefore have to be built from targeted retrieval over
chunks, not from stuffing documents into the context window.

---

## Design decisions

**Chunking follows document structure, not a fixed character count.** Contracts are
organised into numbered articles (`1. ESTABLISHMENT OF DISTRIBUTORSHIP`) and subsections
(`1.1 Grant and Acceptance`). Splitting every N characters cuts clauses in half and leaves
neither fragment able to answer a question. The chunker detects headings by regex and
splits on them; oversized sections fall back to paragraph-boundary splits with a
200-character overlap.

**Overlap is applied only on the fallback path.** When a cut follows a real section
boundary, nothing meaningful is severed and overlap would only duplicate text. When a long
article has to be cut at an arbitrary point, overlap prevents a clause from being lost
across the seam.

**Chunks are cut on the raw text; offsets into the source are preserved.** CUAD annotations
are character offsets into the original string. Normalising whitespace before chunking
shifts every offset and destroys the ground truth. Each chunk therefore stores
`(start_pos, end_pos)` into the unmodified document, and normalisation is applied only to
the copy that gets embedded. This is what makes retrieval measurable rather than
impressionistic.

**Each chunk is prefixed with its document and section name.** A chunk reading *"the term
shall be ten (10) years"* is meaningless in isolation; `[Distributor Agreement — 1.3 Term]
the term shall be ten (10) years` embeds far better. Raw CUAD titles are filenames
(`ReynoldsConsumerProductsInc_20191115_S-1_EX-10.18_11896469_…`), so they are cleaned
before use — 80 characters of identifier noise at the head of every chunk dilutes the
embedding and pulls all chunks from a document toward the same point in vector space.

---

## Results

Measured on 50 contracts, 1,902 chunks, 389 annotated clause spans.
Retrieval is scored within the correct document, isolating ranking quality from document
selection. A hit means the gold span is fully contained in one of the top-k chunks.

### Chunk coverage — the ceiling

| metric | value |
|---|---|
| gold spans fully inside a single chunk | **92.2 %** (1,232 / 1,336) |
| spans split across two chunks | 103 |
| median chunk size | 1,542 chars |

Measured before building retrieval. A clause split across two chunks can never be returned
whole, so no retrieval method can exceed this figure — it is the system's upper bound.

### Retrieval

| engine | R@1 | R@5 | R@10 |
|---|---|---|---|
| vector (pgvector, cosine) | 9.8 % | 31.9 % | 47.8 % |
| lexical (Postgres FTS, `ts_rank`) | 10.0 % | 33.4 % | 47.3 % |
| **hybrid (RRF, k=60)** | **11.1 %** | **33.7 %** | **54.5 %** |

Hybrid search fuses both rankings with Reciprocal Rank Fusion (`score = Σ 1/(60 + rank)`),
which needs no tuning and no score normalisation between engines.

---

## What the experiments showed

**Hybrid search is worth +6.7 points at R@10.** The two engines score almost identically on
their own (47.8 % vs 47.3 %), yet fusing them gains nearly seven points. If they were
retrieving the same chunks, fusion would change nothing — the gain is direct evidence that
they fail on *different* questions. Vector search handles paraphrase; lexical search
catches distinctive legal terms (*indemnify*, *assign*, *governing law*) that embeddings
blur.

**Query phrasing barely matters.** Three formulations were tested — the bare clause label
(`Governing Law`), CUAD's full description, and both concatenated. R@10 ranged from 48.6 %
to 51.2 %, about one point apart on R@5. The bottleneck is not the query, so tuning it
further was abandoned.

**Smaller chunks hurt.** Halving `MAX_CHARS` from 1800 to 700 dropped R@10 from 51 % to
38 %. Part of that is the metric getting stricter — halving chunk size roughly doubles the
candidate pool, so top-10 inspects a smaller share of the document — but the direction was
clear enough to keep the larger size. A fully fair comparison would hold candidate
*coverage* constant rather than k.

**A single malformed token silently broke lexical search.** The first hybrid run scored
1.0 % on the lexical engine. `plainto_tsquery` ANDs all terms, and the question
*"Which state/country's law governs…"* produced the token `'state/country'`, which appears
in no contract — making every query unsatisfiable. Found by inspecting the parsed tsquery
rather than the results. Fixed by extracting keywords with `[a-z]+` (which also splits the
slash), dropping stopwords, deduplicating, and OR-ing the terms with `|`.

---

## Known limitations

- The Parquet mirror used here keeps only positive examples, so the original CUAD
  "no answer" cases are absent. Negatives can be reconstructed: a clause type with no
  annotation for a given contract is, with high probability, genuinely absent from it —
  this will provide the refusal test set.
- Gains concentrate at R@10 (+6.7) and are thin at R@1 (+1.3): fusion pulls more correct
  chunks into the pool without promoting them to the top. This is the signature case for a
  cross-encoder reranker over the top-k candidates.
- Evaluation covers 50 of the 510 contracts.

---

## Roadmap

- [x] Structure-aware chunking with source offsets preserved
- [x] Vector index (pgvector) + retrieval evaluation harness
- [x] Hybrid retrieval (vector + BM25-style FTS, RRF fusion)
- [ ] Grounded generation: mandatory citations, explicit refusal when the answer is absent
- [ ] Faithfulness evaluation + refusal test set built from reconstructed negatives
- [ ] Cross-encoder reranking over top-k
- [ ] Docker Compose for the full stack, CI running the eval suite on every push with a
      recall threshold
- [ ] Latency and cost-per-query tracking

---

## Running it

```bash
docker compose up -d                 # Postgres + pgvector
pip install -r requirements.txt

python src/download.py               # fetch CUAD
python src/explore.py                # corpus statistics
python src/check_chunks.py           # chunk coverage against gold spans
python src/index.py                  # chunk, embed, store
python src/add_fts.py                # full-text index (run AFTER index.py)
python src/eval_hybrid.py            # vector vs lexical vs hybrid
```

`index.py` drops and recreates the table, so `add_fts.py` must run after it.