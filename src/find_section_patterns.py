import json
import re

TARGET = "ENERGOUSCORP_03_16_2017-EX-10.24-STRATEGIC ALLIANCE AGR"

rows = [
    json.loads(line)
    for line in open("../data/cuad_train.jsonl", encoding="utf-8")
]

doc = next(
    r for r in rows
    if r["title"].startswith(TARGET[:40])
)

text = doc["context"]

print(f"Document length: {len(text):,} characters\n")

patterns = {
    "numbered": r"(?<!\w)\d{1,2}\.\s+[A-Z][A-Za-z]",
    "section": r"(?i)(?<!\w)section\s+\d+(?:\.\d+)*",
    "article": r"(?i)(?<!\w)article\s+[IVXLC0-9]+",
}

for name, pattern in patterns.items():
    matches = list(re.finditer(pattern, text))

    print(f"--- {name.upper()} ---")
    print(f"matches: {len(matches)}")

    for m in matches[:15]:
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 120)

        print("\n", repr(text[start:end]))

    print()