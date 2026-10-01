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

def retrieve(conn, question, doc, k=5):
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

SYSTEM = """You extract answers from legal contract passages. You never use outside knowledge.

You receive numbered passages and a question. Produce JSON only:
{"answer": "...", "evidence": "verbatim quote from the cited passage", "citations": [2], "found": true}

RULES
- "evidence" must be copied word-for-word from the passage you cite. Never paraphrase it.
- "citations" lists the passage numbers your evidence comes from.
- Keep "answer" under 50 words.

EXAMPLE 1 — the passages contain the answer:
PASSAGES:
[1] (section: 4. PAYMENT) Payment shall be made within thirty (30) days of invoice.
[2] (section: 9. GOVERNING LAW) This Agreement shall be governed by the laws of the State of Delaware.
QUESTION: Which law governs this agreement?
{"answer": "The laws of the State of Delaware.", "evidence": "This Agreement shall be governed by the laws of the State of Delaware.", "citations": [2], "found": true}

EXAMPLE 2 — the passages do NOT contain the answer:
PASSAGES:
[1] (section: 4. PAYMENT) Payment shall be made within thirty (30) days of invoice.
[2] (section: 9. GOVERNING LAW) This Agreement shall be governed by the laws of the State of Delaware.
QUESTION: What is the minimum purchase commitment?
{"answer": "NOT_FOUND", "evidence": "", "citations": [], "found": false}

Note on Example 2: payment terms are related to purchasing, but they do not state a minimum commitment. Related is not the same as answering. When no passage states the answer outright, return NOT_FOUND.

Return NOT_FOUND whenever you cannot copy a verbatim quote that states the answer. Do not infer, do not reason from similar clauses, do not describe what such contracts usually contain."""

def ask(conn, question, doc, k=5):
    passages = retrieve(conn, question, doc, k)
    if not passages:
        return {"answer": "NOT_FOUND", "citations": [], "found": False}, []

    context = "\n\n".join(
        f"[{i}] (section: {sec})\n{body}"
        for i, (_, sec, _, _, body) in enumerate(passages, start=1))

    try:
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
            options={"temperature": 0, "num_ctx": 8192, "num_predict": 400,
                     "repeat_penalty": 1.1},
        )
    except Exception as e:
        print(f"    [erreur modèle: {type(e).__name__}]")
        return {"answer": "", "citations": [], "found": None, "error": str(e)[:120]}, passages

    raw = resp["message"]["content"].strip()
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