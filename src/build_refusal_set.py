# src/build_refusal_set.py
import json, itertools, random
from collections import defaultdict

random.seed(0)

rows = [json.loads(l) for l in open("../data/cuad_train.jsonl", encoding="utf-8")]
by_doc = defaultdict(list)
for r in rows:
    by_doc[r["title"]].append(r)
indexed = dict(itertools.islice(by_doc.items(), 50))

# le texte de question officiel pour chaque type de clause
question_for = {}
for r in rows:
    question_for.setdefault(r["question_id"],
                            r["question"].split("Details:")[-1].strip())
all_types = set(question_for)

positives, negatives = [], []
for title, anns in indexed.items():
    present = {a["question_id"] for a in anns if a["answer_text"]}
    absent = all_types - present

    for a in anns[:4]:
        if a["answer_text"]:
            positives.append({
                "doc": title, "clause": a["question_id"],
                "question": f'{a["question_id"]}: {question_for[a["question_id"]]}',
                "expected": "FOUND",
                "gold_start": a["answer_start"],
                "gold_end": a["answer_start"] + len(a["answer_text"]),
                "gold_text": a["answer_text"],
            })

    for clause in random.sample(sorted(absent), min(4, len(absent))):
        negatives.append({
            "doc": title, "clause": clause,
            "question": f"{clause}: {question_for[clause]}",
            "expected": "NOT_FOUND",
        })

cases = positives + negatives
with open("../data/testset.jsonl", "w", encoding="utf-8") as f:
    for c in cases:
        f.write(json.dumps(c) + "\n")

print(f"{len(positives)} positifs, {len(negatives)} négatifs — {len(cases)} cas")