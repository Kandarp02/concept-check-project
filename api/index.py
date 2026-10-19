from fastapi import FastAPI

app = FastAPI()

IDEAS = [
    {"id": "backend", "title": "Why does a backend exist at all?"},
    {"id": "api-call", "title": "What does one API call carry?"},
    {"id": "key-browser", "title": "Why can't the key live in the browser?"},
    {"id": "storage", "title": "Why does the app forget without storage?"},
    {"id": "pointer", "title": "Why does the pointer sit on the many side?"},
    {"id": "history", "title": "Why is history a row, not a cell?"},
    {"id": "placement", "title": "How do you decide where a piece lives?"},
]


# Each route is registered twice: with and without the /api prefix.
# Depending on how Vercel forwards the path, the server may see either one.
@app.get("/api/health")
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/ideas")
@app.get("/ideas")
def ideas():
    return IDEAS


# Catch-all: if no route matched, show which path the server received.
@app.get("/{path:path}")
def catch_all(path: str):
    return {"detail": "no route matched", "seen_path": "/" + path}
