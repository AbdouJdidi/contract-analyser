# src/search.py
import sys, psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

DSN = "postgresql://postgres:contract@localhost:5433/contractqa"
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

question = " ".join(sys.argv[1:]) or "What is the term of the agreement?"
qv = model.encode("Represent this sentence for searching relevant passages: " + question,
                  normalize_embeddings=True)

with psycopg.connect(DSN) as conn:
    register_vector(conn)
    res = conn.execute(
        "SELECT doc, section, body, embedding <=> %s AS dist"
        " FROM chunks ORDER BY embedding <=> %s LIMIT 5", (qv, qv)
    ).fetchall()

for doc, section, body, dist in res:
    print(f"\n[{dist:.3f}] {doc} — {section}")
    print(body[:250].replace("\n", " "), "…")