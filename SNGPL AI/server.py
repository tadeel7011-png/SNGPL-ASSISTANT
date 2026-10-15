"""
Free local RAG backend for the Nova frontend.
  Ingest docs : python server.py ingest
  Run server  : uvicorn server:app --port 8000
Put your .txt / .md / .pdf files inside ./docs
"""
import sys, json, glob, os, time, re
import httpx, chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

OLLAMA = "http://localhost:11434"
LLM = "llama3.2"            # any model you pulled with `ollama pull`
EMB = "nomic-embed-text"    # embedding model
TOP_K = 4                   # more context = better answers (lower to 2-3 for speed)

db = chromadb.PersistentClient(path="./db")
col = db.get_or_create_collection("docs")


GREET = {"hi", "hello", "hey", "salam", "salaam", "assalamualaikum", "aoa", "thanks", "thank", "bye", "ok", "okay"}


def is_small_talk(q):
    w = re.sub(r"[^a-z\s]", " ", q.lower()).split()
    return bool(w) and (len(w) <= 3 and w[0] in GREET or " ".join(w) in
                        {"good morning", "good afternoon", "good evening", "who are you",
                         "what can you do", "how are you", "assalam o alaikum"})


def embed(texts):
    out = []
    for i in range(0, len(texts), 16):          # small batches: avoids 400 errors
        batch = [t.replace("\x00", " ").strip() or "." for t in texts[i:i + 16]]
        r = httpx.post(f"{OLLAMA}/api/embed", json={"model": EMB, "input": batch, "keep_alive": "30m"}, timeout=300)
        if r.status_code != 200:
            print("Ollama error:", r.status_code, r.text[:300])
            r.raise_for_status()
        out += r.json()["embeddings"]
    return out


def chunks(text, size=800, overlap=100):
    i = 0
    while i < len(text):
        yield text[i:i + size]
        i += size - overlap


def read_file(path):
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    return open(path, encoding="utf-8", errors="ignore").read()


def ingest():
    for p in glob.glob("docs/**/*", recursive=True):
        if not p.lower().endswith((".txt", ".md", ".pdf")):
            continue
        cs = [c for c in chunks(read_file(p)) if c.strip()]
        if not cs:
            continue
        for i in range(0, len(cs), 64):
            part = cs[i:i + 64]
            col.upsert(
                ids=[f"{p}:{i + j}" for j in range(len(part))],
                documents=part,
                embeddings=embed(part),
                metadatas=[{"title": os.path.basename(p), "chunk": i + j} for j in range(len(part))],
            )
        print("indexed", p, len(cs), "chunks")


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/v1/chat/completions")
async def chat(body: dict):
    msgs = body["messages"]
    T0 = time.time()
    q = next(m["content"] for m in reversed(msgs) if m["role"] == "user")

    # short follow-ups ("tell me more", "and for managers?") lose meaning alone, so search with the previous question too
    users = [m["content"] for m in msgs if m["role"] == "user"]
    search_q = q if (len(q.split()) >= 6 or len(users) < 2) else users[-2] + " " + q

    small = is_small_talk(q)
    if small:
        docs, metas = [], []
    elif col.count():
        hits = col.query(query_embeddings=embed([search_q]), n_results=TOP_K)
        docs, metas = hits["documents"][0], hits["metadatas"][0]
    else:
        docs, metas = [], []

    print(f"[timing] search (embed+vector db): {time.time() - T0:.1f}s")
    ctx = "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs))
    seen, sources = set(), []
    for m in metas:
        if m["title"] not in seen:
            seen.add(m["title"])
            sources.append({"title": m["title"]})

    system = msgs[0]["content"] if msgs and msgs[0]["role"] == "system" else ""
    if small:
        sys_prompt = (
            "You are SNGPL AI, the internal assistant of SNGPL (Sui Northern Gas Pipelines Limited). "
            "The user sent a greeting or small talk. Reply warmly in 1 or 2 short sentences, and say you can "
            "answer questions from the SNGPL manuals (Admin, Customer Service, HR, HSE, Medical Policy, Risk Management)."
        )
    else:
        sys_prompt = (
            "You are SNGPL AI, the internal assistant of SNGPL (Sui Northern Gas Pipelines Limited), "
            "Pakistan's gas transmission and distribution company headquartered in Lahore. "
            "SNGPL ALWAYS means Sui Northern Gas Pipelines Limited. Never invent another meaning. "
            "Answer ONLY from the context below. If the context does not contain the answer, reply: "
            "'I could not find this in the SNGPL manuals.' Do not guess or use outside knowledge. "
            "Be concise: 3 to 5 short sentences.\n\nContext:\n" + ctx
        )
    prompt = [{"role": "system", "content": sys_prompt}] + [m for m in msgs if m["role"] != "system"]

    async def gen():
        gen.first = False
        if sources:
            yield "data: " + json.dumps({"sources": sources}) + "\n\n"
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", f"{OLLAMA}/api/chat",
                                json={"model": body.get("model", LLM), "messages": prompt, "stream": True,
                                      "keep_alive": "30m", "options": {"num_ctx": 3072, "num_predict": 350}}) as r:
                async for line in r.aiter_lines():
                    if not line:
                        continue
                    j = json.loads(line)
                    if j.get("error"):
                        print("[ollama error]", j["error"])
                        msg = "Ollama error: " + str(j["error"])
                        yield "data: " + json.dumps({"choices": [{"delta": {"content": msg}}]}) + "\n\n"
                        break
                    t = j.get("message", {}).get("content", "")
                    if t and not getattr(gen, 'first', False):
                        gen.first = True
                        print(f"[timing] first token after: {time.time() - T0:.1f}s")
                    if t:
                        yield "data: " + json.dumps({"choices": [{"delta": {"content": t}}]}) + "\n\n"
                    if j.get("done"):
                        print(f"[timing] total: {time.time() - T0:.1f}s")
                        break
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest()
    else:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)