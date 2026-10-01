# src/check_claim.py
import psycopg
DSN = "postgresql://postgres:contract@localhost:5433/contractqa"
with psycopg.connect(DSN) as conn:
    doc = conn.execute("SELECT DISTINCT doc FROM chunks LIMIT 1").fetchone()[0]
    rows = conn.execute(
        "SELECT id, start_pos, body FROM chunks WHERE doc = %s AND body ILIKE %s",
        (doc, "%New York%")).fetchall()
    print(f"{len(rows)} chunks mentionnent New York")
    for cid, s, body in rows[:3]:
        i = body.lower().find("new york")
        print(f"\nchunk {cid} (char {s}): …{body[max(0,i-200):i+150]}…")