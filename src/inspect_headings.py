# src/inspect_headings.py
import json, re
from collections import defaultdict

TARGET = "ENERGOUSCORP_03_16_2017-EX-10.24-STRATEGIC ALLIANCE AGR"
rows = [json.loads(l) for l in open("../data/cuad_train.jsonl", encoding="utf-8")]
by_doc = defaultdict(list)
for r in rows:
    by_doc[r["title"]].append(r)

title = next(t for t in by_doc if t.startswith(TARGET[:40]))
text = by_doc[title][0]["context"]

print(f"--- {title[:60]} ---\n")
for i, line in enumerate(text.split("\n")[:120]):
    s = line.strip()
    if not s:
        continue
    # lignes qui ressemblent à un titre : courtes, ou commençant par un numéro/ARTICLE/SECTION
    if (len(s) < 80 and (s.isupper() or re.match(r"^(ARTICLE|SECTION|\d+[\.\)]|\([a-z0-9]+\))", s, re.I))):
        print(f"{i:4d} | {line[:100]}")
    # add at the end of inspect_headings.py
lines = text.split("\n")
print(f"\nlongueur: {len(text):,} car., {len(lines)} lignes")
print(f"longueur médiane de ligne: {sorted(len(l) for l in lines)[len(lines)//2]}")
print("\n--- 1500 premiers caractères bruts ---")
print(repr(text[:1500]))