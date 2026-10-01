import json
rows = [json.loads(l) for l in open("../data/cuad_train.jsonl", encoding="utf-8")]
texte = rows[0]["context"]
open("../data/sample_contract.txt", "w", encoding="utf-8").write(texte)
print(texte[:3000])