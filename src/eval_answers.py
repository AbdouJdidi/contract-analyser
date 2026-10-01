# src/eval_answers.py
import json, time
import psycopg
from pgvector.psycopg import register_vector
from answer import ask, retrieve

DSN = "postgresql://postgres:contract@localhost:5433/contractqa"   # adapte le port
LIMIT = 60

cases = [json.loads(l) for l in open("../data/testset.jsonl", encoding="utf-8")]
if LIMIT:
    cases = cases[:LIMIT//2] + cases[len(cases)//2:len(cases)//2 + LIMIT//2]

verified=0
pos = neg = 0
answered_pos = refused_neg = 0
grounded = retrievable = 0
cited_ok = 0
errors=0
t0 = time.time()

with psycopg.connect(DSN) as conn:
    register_vector(conn)
    for i, c in enumerate(cases, 1):
        result, passages = ask(conn, c["question"], c["doc"], k=10)
        found = result.get("found") is True
        if result.get("found") is None:
            errors += 1

        if c["expected"] == "FOUND":
            pos += 1
            hit = any(sp <= c["gold_start"] and c["gold_end"] <= ep
                      for _, _, sp, ep, _ in passages)
            retrievable += hit
            if found:
                answered_pos += 1
                for n in result.get("citations", []):
                    if 1 <= n <= len(passages):
                        _, _, sp, ep, _ = passages[n - 1]
                        if sp <= c["gold_start"] and c["gold_end"] <= ep:
                            cited_ok += 1
                            break
                if hit:
                    grounded += 1

                # la citation est-elle vérifiable textuellement ?
                ev = (result.get("evidence") or "").strip()
                if ev:
                    cited_bodies = " ".join(
                        passages[n-1][4] for n in result.get("citations", [])
                        if 1 <= n <= len(passages))
                    norm = lambda s: " ".join(s.lower().split())
                    if norm(ev[:120]) in norm(cited_bodies):
                        verified += 1
        else:
            neg += 1
            if not found:
                refused_neg += 1

        if i % 25 == 0:
            print(f"  {i}/{len(cases)}  ({time.time()-t0:.0f}s)")


print(f"\n--- {pos} positifs, {neg} négatifs, {time.time()-t0:.0f}s ---\n")
print(f"rappel du retrieval (clause récupérée)   : {retrievable/pos:.1%}")
print(f"taux de réponse sur positifs             : {answered_pos/pos:.1%}")
print(f"  dont la clause était bien récupérée    : {grounded/max(answered_pos,1):.1%}")
print(f"  dont la citation pointe la bonne clause: {cited_ok/max(answered_pos,1):.1%}")
print(f"  dont l'evidence est vérifiable littéralement: {verified/max(answered_pos,1):.1%}")
print(f"taux de refus correct sur négatifs       : {refused_neg/neg:.1%}")
print(f"erreurs modèle : {errors}/{len(cases)}")