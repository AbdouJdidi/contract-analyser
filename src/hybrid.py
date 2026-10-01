def rrf(rankings, k=60):
    """rankings: list of lists of chunk ids, best first."""
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

def search(conn, model, question, doc=None, limit=10, prefix=
           "Represent this sentence for searching relevant passages: "):
    qv = model.encode(prefix + question, normalize_embeddings=True)
    where = "WHERE doc = %s" if doc else ""
    args_v = (qv, doc, qv) if doc else (qv, qv)

    vec = [r[0] for r in conn.execute(
        f"SELECT id FROM chunks {where} ORDER BY embedding <=> %s LIMIT 30",
        (doc, qv) if doc else (qv,)).fetchall()]

    lex = [r[0] for r in conn.execute(
        f"SELECT id FROM chunks {'WHERE doc = %s AND' if doc else 'WHERE'}"
        f" tsv @@ plainto_tsquery('english', %s)"
        f" ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC LIMIT 30",
        (doc, question, question) if doc else (question, question)).fetchall()]

    return rrf([vec, lex])[:limit]