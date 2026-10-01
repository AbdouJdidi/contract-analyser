import json, statistics
from collections import Counter

rows = [json.loads(l) for l in open("../data/cuad_train.jsonl", encoding="utf-8")]

print("colonnes:", list(rows[0].keys()))
print("lignes:", len(rows))

contrats = {r.get("title") or r["context"][:200]: r["context"] for r in rows}
print("contrats uniques:", len(contrats))

vides = sum(1 for r in rows if not r["answer_text"])
print(f"sans réponse: {vides} ({vides/len(rows):.0%})")

longueurs = [len(c) for c in contrats.values()]
print(f"taille des contrats: médiane {statistics.median(longueurs):,.0f} car., "
      f"max {max(longueurs):,.0f} car.")

print("\ntypes de clauses:")
for q, n in Counter(r["question_id"] for r in rows).most_common(15):
    print(f"  {n:6d}  {q}")