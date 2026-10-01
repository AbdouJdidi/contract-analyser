# src/download.py
from datasets import load_dataset

ds = load_dataset("Nadav-Timor/CUAD")
print(ds)
print("colonnes:", ds["train"].column_names)
print(ds["train"][0])

ds["train"].to_json("../data/cuad_train.jsonl")