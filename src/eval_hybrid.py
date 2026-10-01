# src/eval_hybrid.py
import json, itertools
from collections import defaultdict
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
import re

DSN = "postgresql://postgres:contract@localhost:5433/contractqa"   # adapte le port
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
PREFIX = "Represent this sentence for searching relevant passages: "
POOL = 30

rows = [json.loads(l) for l in open("../data/cuad_train.jsonl", encoding="utf-8")]
by_doc = defaultdict(list)
for r in rows:
    by_doc[r["title"]].append(r)
indexed = dict(itertools.islice(by_doc.items(), 50))

def vector_ids(conn, doc, qv):
    return [r[0] for r in conn.execute(
        "SELECT id FROM chunks WHERE doc = %s ORDER BY embedding <=> %s LIMIT %s",
        (doc, qv, POOL)).fetchall()]

STOP = {"the","of","a","an","any","which","what","this","that","is","are","to","in",
        "for","and","or","be","by","with","on","as","if","it","its","will","shall"}

def keywords(q):
    words = re.findall(r"[a-z]+", q.lower())
    kept = [w for w in words if w not in STOP and len(w) > 2]
    return " | ".join(dict.fromkeys(kept))      

def lexical_ids(conn, doc, q):
    kw = keywords(q)
    if not kw:
        return []
    return [r[0] for r in conn.execute(
        "SELECT id FROM chunks WHERE doc = %s"
        " AND tsv @@ to_tsquery('english', %s)"
        " ORDER BY ts_rank(tsv, to_tsquery('english', %s)) DESC LIMIT %s",
        (doc, kw, kw, POOL)).fetchall()]

def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

def covers(conn, ids, s, e):
    if not ids:
        return False
    res = conn.execute(
        "SELECT start_pos, end_pos FROM chunks WHERE id = ANY(%s)", (ids,)).fetchall()
    return any(sp <= s and e <= ep for sp, ep in res)

results = {name: {1: 0, 5: 0, 10: 0} for name in ("vectoriel", "lexical", "hybride")}
total = 0

with psycopg.connect(DSN) as conn:
    register_vector(conn)
    for title, anns in indexed.items():
        for a in anns[:8]:
            if not a["answer_text"]:
                continue
            q = f'{a["question_id"]}: {a["question"].split("Details:")[-1].strip()}'
            s, e = a["answer_start"], a["answer_start"] + len(a["answer_text"])
            qv = model.encode(PREFIX + q, normalize_embeddings=True)

            vec = vector_ids(conn, title, qv)
            lex = lexical_ids(conn, title, q)
            hyb = rrf([vec, lex])

            total += 1
            for name, ids in (("vectoriel", vec), ("lexical", lex), ("hybride", hyb)):
                for k in (1, 5, 10):
                    if covers(conn, ids[:k], s, e):
                        results[name][k] += 1

print(f"{total} questions\n")
print(f"{'moteur':12s} {'R@1':>7s} {'R@5':>7s} {'R@10':>7s}")
for name, h in results.items():
    print(f"{name:12s} {h[1]/total:6.1%} {h[5]/total:6.1%} {h[10]/total:6.1%}")