import time, psycopg
from pgvector.psycopg import register_vector
from answer import ask

DSN = "postgresql://postgres:contract@localhost:5433/contractqa"   # adapte le port
QS = ["What is the governing law of this contract?",
      "Can this agreement be assigned to a third party?",
      "What is the termination notice period?"]

with psycopg.connect(DSN) as conn:
    register_vector(conn)
    doc = conn.execute("SELECT DISTINCT doc FROM chunks LIMIT 1").fetchone()[0]
    for i, q in enumerate(QS, 1):
        t = time.time()
        r, _ = ask(conn, q, doc)
        print(f"{i}. {time.time()-t:5.1f}s  found={r.get('found')}  {r.get('answer','')[:70]}")