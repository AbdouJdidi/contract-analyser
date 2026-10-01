# src/eval_retrieval.py
import json, itertools
from collections import defaultdict
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from chunker import chunk_contract

DSN = "postgresql://postgres:contract@localhost:5433/contractqa"  # adapte le port
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
PREFIX = "Represent this sentence for searching relevant passages: "

rows = [json.loads(l) for l in open("../data/cuad_train.jsonl", encoding="utf-8")]
by_doc = defaultdict(list)
for r in rows:
    by_doc[r["title"]].append(r)
indexed = dict(itertools.islice(by_doc.items(), 50))

K = 5
hits = total = 0
with psycopg.connect(DSN) as conn:
    register_vector(conn)
    for title, anns in indexed.items():
        for a in anns[:8]:                      # échantillon par contrat
            if not a["answer_text"]:
                continue
            s, e = a["answer_start"], a["answer_start"] + len(a["answer_text"])
            qv = model.encode(PREFIX + a["question_id"], normalize_embeddings=True)
            res = conn.execute(
                "SELECT doc, start_pos, end_pos FROM chunks"
                " WHERE doc = %s ORDER BY embedding <=> %s LIMIT %s",
                (title, qv, K),
            ).fetchall()
            total += 1
            if any(sp <= s and e <= ep for _, sp, ep in res):
                hits += 1

print(f"recall@{K}: {hits}/{total} = {hits/total:.1%}")