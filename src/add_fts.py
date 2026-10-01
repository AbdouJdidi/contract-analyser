import psycopg
DSN = "postgresql://postgres:contract@localhost:5433/contractqa"   # adapte le port
with psycopg.connect(DSN) as conn:
    conn.execute("ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector")
    conn.execute("UPDATE chunks SET tsv = to_tsvector('english', body)")
    conn.execute("CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING GIN (tsv)")
    conn.commit()
print("full-text index ready")