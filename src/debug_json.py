# src/debug_min.py
from answer import SYSTEM, MODEL_LLM, ollama_client

def try_it(label, **kw):
    r = ollama_client.chat(model=MODEL_LLM, keep_alive="30m",
                           options={"temperature": 0, "num_ctx": 8192, "num_predict": 300},
                           **kw)
    txt = r["message"]["content"]
    print(f"{label:28s} -> {len(txt):4d} chars  {repr(txt[:120])}")
    return r

# 1. aucun system prompt, question courte
try_it("user seul", messages=[{"role": "user", "content": "What is 2+2?"}])

# 2. system prompt seul
try_it("system + question courte",
       messages=[{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "What is 2+2?"}])

# 3. system + un faux passage court
fake = "[1] (section: 10. GOVERNING LAW)\nThis Agreement shall be governed by the laws of the State of New York."
try_it("system + passage court",
       messages=[{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": f"PASSAGES:\n{fake}\n\nQUESTION: What is the governing law?"}])