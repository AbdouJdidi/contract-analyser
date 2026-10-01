import json, statistics
from collections import defaultdict
from chunker import chunk_contract

rows = [json.loads(l) for l in open("../data/cuad_train.jsonl", encoding="utf-8")]

by_doc = defaultdict(list)
for r in rows:
    by_doc[r["title"]].append(r)

total = covered = split_across = 0
sizes = []

for title, anns in list(by_doc.items())[:50]:      # 50 contracts is enough to judge
    text = anns[0]["context"]
    chunks = chunk_contract(text, title)
    sizes += [len(c["text"]) for c in chunks]

    for a in anns:
        if not a["answer_text"]:
            continue
        s = a["answer_start"]
        e = s + len(a["answer_text"])
        total += 1
        hits = [c for c in chunks if c["start"] <= s and e <= c["end"]]
        if hits:
            covered += 1
        elif any(c["start"] < e and s < c["end"] for c in chunks):
            split_across += 1

print(f"chunks: {len(sizes)}, median size {statistics.median(sizes):.0f} chars")
print(f"gold spans fully inside one chunk: {covered}/{total} ({covered/total:.1%})")
print(f"spans cut across two chunks: {split_across}")