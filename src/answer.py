# src/answer.py
import json, re, sys
import psycopg
from ollama import Client
from sentence_transformers import SentenceTransformer
from pgvector.psycopg import register_vector

ollama_client = Client(timeout=180)

DSN = "postgresql://postgres:contract@localhost:5433/contractqa"   # adapte le port
MODEL_LLM = "qwen3:4b"
POOL = 30

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

STOP = {"the","of","a","an","any","which","what","this","that","is","are","to","in",
        "for","and","or","be","by","with","on","as","if","it","its","will","shall"}

def keywords(q):
    words = re.findall(r"[a-z]+", q.lower())
    kept = [w for w in words if w not in STOP and len(w) > 2]
    return " | ".join(dict.fromkeys(kept))

def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0) + 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)

def retrieve(conn, question, doc, k=3):
    qv = model.encode("Represent this sentence for searching relevant passages: " + question,
                      normalize_embeddings=True)
    vec = [r[0] for r in conn.execute(
        "SELECT id FROM chunks WHERE doc = %s ORDER BY embedding <=> %s LIMIT %s",
        (doc, qv, POOL)).fetchall()]
    kw = keywords(question)
    lex = [r[0] for r in conn.execute(
        "SELECT id FROM chunks WHERE doc = %s AND tsv @@ to_tsquery('english', %s)"
        " ORDER BY ts_rank(tsv, to_tsquery('english', %s)) DESC LIMIT %s",
        (doc, kw, kw, POOL)).fetchall()] if kw else []

    ids = rrf([vec, lex])[:k]
    if not ids:
        return []
    rows = conn.execute(
        "SELECT id, section, start_pos, end_pos, body FROM chunks WHERE id = ANY(%s)",
        (ids,)).fetchall()
    order = {cid: i for i, cid in enumerate(ids)}
    rows.sort(key=lambda r: order[r[0]])
    return rows

SYSTEM = """You answer questions about legal contracts.

Rules, in order of priority:
1. Use ONLY the numbered passages provided. Never use outside knowledge about contracts or law.
2. Every factual statement must cite the passage it comes from, as [1], [2]. A statement without a citation is a failure.
3. If the passages do not contain the answer, reply exactly: NOT_FOUND. Do not guess, do not infer from similar clauses, do not say what is "typical" in such contracts.
4. Quote the operative wording where it matters, but keep the answer under 80 words.

Return JSON only:
{"answer": "...", "citations": [1, 3], "found": true}
If the answer is absent: {"answer": "NOT_FOUND", "citations": [], "found": false}"""

def ask(conn, question, doc, k=3):
    passages = retrieve(conn, question, doc, k)
    if not passages:
        return {"answer": "NOT_FOUND", "citations": [], "found": False}, []

    context = "\n\n".join(
        f"[{i}] (section: {sec})\n{body}"
        for i, (_, sec, _, _, body) in enumerate(passages, start=1))

    resp = ollama_client.chat(
        model=MODEL_LLM,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"PASSAGES:\n{context}\n\nQUESTION: {question}"},
        ],
        format="json",
        think=False,
        keep_alive="30m",
        options={"temperature": 0, "num_ctx": 4096, "num_predict": 200},
    )
    raw = resp["message"]["content"].strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    try:
        return json.loads(raw), passages
    except json.JSONDecodeError:
        return {"answer": raw, "citations": [], "found": None}, passages

if __name__ == "__main__":

    with psycopg.connect(DSN) as conn:
        register_vector(conn)

        doc = conn.execute(
            "SELECT DISTINCT doc FROM chunks LIMIT 1"
        ).fetchone()[0]

        question = " ".join(sys.argv[1:]) or \
            "What is the governing law of this contract?"

        result, passages = ask(conn, question, doc)

        print(f"DOC: {doc[:60]}")
        print(f"Q: {question}")

        print("ANSWER:", result["answer"])
        print("FOUND :", result["found"])

        for n in result.get("citations", []):
            if 1 <= n <= len(passages):
                _, sec, s, e, body = passages[n - 1]
                print(f"\n  [{n}] {sec}  (chars {s}-{e})")
                # montre la zone qui recoupe la réponse plutôt que le début du passage
                ans_words = [w for w in re.findall(r"[A-Za-z]{4,}", result["answer"])]
                pos = next((body.find(w) for w in ans_words if body.find(w) != -1), 0)
                lo = max(0, pos - 150)
                print("     …", body[lo:pos + 250].replace("\n", " "), "…")

        print("\n--- all retrieved passages ---")

        for i, (_, sec, s, e, body) in enumerate(passages, start=1):
            mark = "*" if i in result.get("citations", []) else " "

            print(f"\n{mark}[{i}] {sec} (chars {s}-{e})")
            print("   ", body[:300].replace("\n", " "), "...")