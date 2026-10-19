from urllib.parse import parse_qs

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


class RestorePath:
    """Vercel's rewrite sends every /api/... request to /api/index and
    loses the original path. vercel.json passes it along as ?p=<path>,
    and this wrapper puts it back before FastAPI picks a route."""

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"] == "/api/index":
            query = parse_qs(scope.get("query_string", b"").decode())
            if "p" in query:
                scope = dict(scope)
                scope["path"] = "/api/" + query["p"][0].lstrip("/")
        await self.inner(scope, receive, send)


# Vercel looks for the variable called `app`, so the wrapper takes that name.
fastapi_app = app
app = RestorePath(fastapi_app)