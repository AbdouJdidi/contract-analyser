# src/eval_queries.py
import json, itertools
from collections import defaultdict
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:contract@localhost:5433/contractqa"   # adapte le port
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
PREFIX = "Represent this sentence for searching relevant passages: "

rows = [json.loads(l) for l in open("../data/cuad_train.jsonl", encoding="utf-8")]
by_doc = defaultdict(list)
for r in rows:
    by_doc[r["title"]].append(r)
indexed = dict(itertools.islice(by_doc.items(), 50))

def q_id(a):      return a["question_id"]
def q_details(a): return a["question"].split("Details:")[-1].strip()
def q_both(a):    return f'{a["question_id"]}: {a["question"].split("Details:")[-1].strip()}'

VARIANTS = {"id seul": q_id, "details": q_details, "id + details": q_both}

with psycopg.connect(DSN) as conn:
    register_vector(conn)
    for name, build in VARIANTS.items():
        hits = {1: 0, 5: 0, 10: 0}
        total = 0
        for title, anns in indexed.items():
            for a in anns[:8]:
                if not a["answer_text"]:
                    continue
                s, e = a["answer_start"], a["answer_start"] + len(a["answer_text"])
                qv = model.encode(PREFIX + build(a), normalize_embeddings=True)
                res = conn.execute(
                    "SELECT start_pos, end_pos FROM chunks WHERE doc = %s"
                    " ORDER BY embedding <=> %s LIMIT 10", (title, qv)
                ).fetchall()
                total += 1
                for k in hits:
                    if any(sp <= s and e <= ep for sp, ep in res[:k]):
                        hits[k] += 1
        print(f"{name:14s}  R@1 {hits[1]/total:5.1%}   R@5 {hits[5]/total:5.1%}   R@10 {hits[10]/total:5.1%}")