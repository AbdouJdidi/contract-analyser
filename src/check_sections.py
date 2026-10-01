# src/check_sections.py
import json, itertools
from collections import defaultdict, Counter
from chunker import chunk_contract

rows = [json.loads(l) for l in open("../data/cuad_train.jsonl", encoding="utf-8")]
by_doc = defaultdict(list)
for r in rows:
    by_doc[r["title"]].append(r)

bad = 0
for title, anns in itertools.islice(by_doc.items(), 50):
    chunks = chunk_contract(anns[0]["context"], title)
    labels = Counter(c["section"] for c in chunks)
    if len(labels) <= 2:
        bad += 1
        print(f"{len(labels):2d} sections — {title[:55]}")
print(f"\n{bad}/50 contrats sans structure détectée")