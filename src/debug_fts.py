import psycopg
DSN = "postgresql://postgres:contract@localhost:5433/contractqa"

with psycopg.connect(DSN) as conn:
    n, filled = conn.execute(
        "SELECT count(*), count(tsv) FROM chunks").fetchone()
    print(f"chunks: {n}, avec tsv: {filled}")

    q = "Governing Law: Which state/country's law governs the interpretation of the contract?"
    print("plainto:", conn.execute("SELECT plainto_tsquery('english', %s)::text", (q,)).fetchone()[0][:200])
    print("websearch:", conn.execute("SELECT websearch_to_tsquery('english', %s)::text", (q,)).fetchone()[0][:200])

    for fn in ("plainto_tsquery", "websearch_to_tsquery"):
        c = conn.execute(
            f"SELECT count(*) FROM chunks WHERE tsv @@ {fn}('english', %s)", (q,)).fetchone()[0]
        print(f"{fn}: {c} chunks matched")