# src/index.py
import json, itertools
from collections import defaultdict
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer
from chunker import chunk_contract

DSN = "postgresql://postgres:contract@localhost:5433/contractqa"
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

rows = [json.loads(l) for l in open("../data/cuad_train.jsonl", encoding="utf-8")]
by_doc = defaultdict(list)
for r in rows:
    by_doc[r["title"]].append(r)

chunks = []
for title, anns in itertools.islice(by_doc.items(), 50):   # 50 contrats pour commencer
    chunks += chunk_contract(anns[0]["context"], title)
print(f"{len(chunks)} chunks à indexer")

vecs = model.encode([c["text"] for c in chunks],
                    batch_size=32, normalize_embeddings=True, show_progress_bar=True)

with psycopg.connect(DSN) as conn:
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("""
        CREATE TABLE chunks (
            id        bigserial PRIMARY KEY,
            doc       text NOT NULL,
            section   text NOT NULL,
            start_pos int  NOT NULL,
            end_pos   int  NOT NULL,
            body      text NOT NULL,
            embedding vector(384)
        )
    """)
    with conn.cursor() as cur:
        for c, v in zip(chunks, vecs):
            cur.execute(
                "INSERT INTO chunks (doc, section, start_pos, end_pos, body, embedding)"
                " VALUES (%s,%s,%s,%s,%s,%s)",
                (c["doc"], c["section"], c["start"], c["end"], c["text"], v),
            )
    conn.commit()
print("indexé")