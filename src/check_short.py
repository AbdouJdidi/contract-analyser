# src/check_short.py
import json, itertools, statistics
from collections import defaultdict
from chunker import chunk_contract

rows = [json.loads(l) for l in open("../data/cuad_train.jsonl", encoding="utf-8")]
by_doc = defaultdict(list)
for r in rows:
    by_doc[r["title"]].append(r)

flat, structured = [], []
for title, anns in itertools.islice(by_doc.items(), 50):
    text = anns[0]["context"]
    n_sections = len({c["section"] for c in chunk_contract(text, title)})
    (flat if n_sections <= 2 else structured).append(len(text))

print(f"sans structure : {len(flat):2d} contrats, taille médiane {statistics.median(flat):,.0f} car.")
print(f"avec structure : {len(structured):2d} contrats, taille médiane {statistics.median(structured):,.0f} car.")