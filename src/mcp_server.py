#!/usr/bin/env python3
"""
MCP server over the drivebot retrieval pipeline.

Exposes the same grounded retrieval the Slack bot uses to any MCP client
(Claude Desktop, Cursor, Claude Code) over stdio.

One design decision worth stating up front, because it is the whole point:

    answer_question does NOT call an LLM.

The MCP client is already a model. Calling a second one here would add a
round-trip, a bill, and a second place for the answer to drift. So this server
runs Gate 1 -- the deterministic half, the one that actually prevents
hallucination -- and hands the client either a refusal or numbered sources with
the citation rules attached. The client model becomes Gate 2.

The consequence is the interesting bit: ask an MCP client something the corpus
has never heard of, and instead of a confident invention it gets

    REFUSED (gate: unknown-term)

Run:
    pip install "mcp[cli]"
    cd src && python3 mcp_server.py

Debug without a client:
    npx @modelcontextprotocol/inspector python3 mcp_server.py
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
from urllib.parse import parse_qs
from pathlib import Path

# The package uses flat imports (`from chunking import Chunk`), so src/ has to
# be importable regardless of the cwd the MCP client launches us from. Clients
# do not inherit your shell.
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# stdout is the MCP protocol channel. Anything printed there corrupts the
# stream and the client disconnects with a parse error that names no cause.
# Every log line goes to stderr, without exception.
logging.basicConfig(
    level=os.environ.get("MCP_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s drivebot.mcp: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("drivebot.mcp")

# The rest of the package reads os.environ directly and relies on systemd's
# EnvironmentFile= to fill it. An MCP client is not systemd -- it spawns this
# process with a bare environment -- so .env has to be loaded here, or hybrid
# dies on a missing GEMINI_API_KEY.
#
# override=False on purpose: anything the client passes in its own `env` block
# wins over the file, so a config can still switch backends per connection.
try:
    from dotenv import load_dotenv

    if (ROOT / ".env").is_file():
        load_dotenv(ROOT / ".env", override=False)
        log.info("loaded %s", ROOT / ".env")
except ImportError:  # pragma: no cover - python-dotenv ships with mcp[cli]
    log.warning("python-dotenv not installed; relying on the ambient environment")

from mcp.server.fastmcp import FastMCP  # noqa: E402

from answering import REFUSAL, answer, build_context  # noqa: E402
from cli import load_corpus  # noqa: E402  (has a __main__ guard; safe to import)
from config import Settings  # noqa: E402
from retrieval import build_retriever  # noqa: E402

MAX_K = 20

# stdio is the default because that is what a desktop client spawns directly.
# streamable-http exists for the case where the client cannot spawn a local
# process at all -- a phone, a browser, a sandboxed desktop app, or a client
# on someone else's machine. Behind a tunnel it is also the only form of this
# server you can hand to another person as a URL.
TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")
HOST = os.environ.get("MCP_HOST", "127.0.0.1")
PORT = int(os.environ.get("MCP_PORT", "8765"))

# Only enforced on http. stdio needs no token: the client spawned the process
# itself, so it already has whatever access the token would have granted.
TOKEN = os.environ.get("MCP_TOKEN", "")

CITATION_RULES = (
    "Answer the user strictly from the sources above.\n"
    "- Cite the source number inline for every claim, like [1].\n"
    "- If the sources only partly answer the question, answer that part and "
    "say plainly which part they do not cover.\n"
    "- Do not fill gaps from your own knowledge, and do not reason from what "
    "is usually true. These documents are the only authority."
)

# The SDK checks the Host header against an allowlist (DNS-rebinding
# protection) and answers 421 to anything unlisted. Localhost is fine by
# default, but behind a tunnel the Host becomes the public domain -- so it has
# to be named explicitly. Set MCP_PUBLIC_HOST to the tunnel hostname, without
# the scheme: MCP_PUBLIC_HOST=something.trycloudflare.com
PUBLIC_HOST = os.environ.get("MCP_PUBLIC_HOST", "").strip()

_security = None
if PUBLIC_HOST:
    from mcp.server.transport_security import TransportSecuritySettings

    _security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[PUBLIC_HOST, f"{HOST}:{PORT}", "127.0.0.1:*", "localhost:*"],
        allowed_origins=[f"https://{PUBLIC_HOST}", f"http://{PUBLIC_HOST}"],
    )

mcp = FastMCP(
    name="drivebot",
    host=HOST,
    port=PORT,
    transport_security=_security,
    instructions=(
        "Grounded retrieval over an indexed document corpus.\n\n"
        "Call answer_question for user questions: it applies a deterministic "
        "grounding gate and returns either the relevant sources or an explicit "
        "refusal. Call search_documents when you want ranked passages without "
        "the gate, and list_documents to see what is indexed.\n\n"
        "When a tool returns REFUSED, the corpus does not contain the answer. "
        "Report that to the user. Do NOT substitute your own knowledge of the "
        "topic -- a plausible answer that the documents do not support is the "
        "exact failure this server exists to prevent."
    ),
)


# ---------------------------------------------------------------------------
# Lazy singleton.
#
# Building the hybrid backend embeds the whole corpus. Doing that at import
# time would stall the MCP handshake and some clients time out and kill the
# process before it ever answers. Build on first use instead.
# ---------------------------------------------------------------------------

_state: dict = {}


def _corpus_dir() -> Path:
    raw = os.environ.get("CORPUS_DIR", "corpus")
    p = Path(raw)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def _ready() -> tuple:
    if "retriever" not in _state:
        corpus = _corpus_dir()
        chunks, manifest = load_corpus(corpus)
        settings = Settings.from_env()
        retriever = build_retriever(settings, chunks)
        _state.update(retriever=retriever, manifest=manifest, chunks=chunks)
        log.info(
            "ready: backend=%s corpus=%s chunks=%d",
            retriever.name,
            corpus,
            len(chunks),
        )
    return _state["retriever"], _state["manifest"]


def _clamp(k: int, default: int) -> int:
    try:
        return max(1, min(int(k), MAX_K))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def answer_question(question: str, k: int = 4) -> str:
    """Retrieve grounded sources for a question, or refuse if the corpus cannot support an answer.

    This is the tool to use for user questions. It applies a deterministic
    grounding gate before returning anything: if the corpus has never seen the
    key term in the question, or if no retrieved passage actually covers it,
    the tool refuses rather than returning loosely-related text.

    A REFUSED result means the answer is not in the documents. Report the
    refusal to the user instead of answering from your own knowledge.

    Args:
        question: The user's question, in natural language.
        k: How many passages to retrieve (1-20).

    Returns:
        Numbered sources with citation rules, or an explicit refusal naming
        which gate rejected the question.
    """
    question = (question or "").strip()
    if not question:
        return "ERROR: question must not be empty."

    try:
        retriever, _ = _ready()
        # call_model=False runs both gates and stops. The client is the model.
        res = answer(question, retriever, k=_clamp(k, 4), call_model=False)
    except Exception as exc:
        log.exception("answer_question failed")
        return f"ERROR: {exc}"

    if res.refused:
        log.info("REFUSED gate=%s q=%r", res.gate, question)
        return (
            f"REFUSED (gate: {res.gate})\n\n{REFUSAL}\n\n"
            "The corpus does not support an answer to this question. "
            "Tell the user this. Do not answer from your own knowledge."
        )

    log.info(
        "answered q=%r hits=%d top=%.2f",
        question,
        len(res.hits),
        res.hits[0].score if res.hits else 0.0,
    )
    return f"SOURCES:\n\n{build_context(res.hits)}\n\n---\n\n{CITATION_RULES}"


@mcp.tool()
def search_documents(query: str, k: int = 5) -> str:
    """Return ranked passages from the corpus, without the grounding gate.

    Use this for exploration -- browsing what the corpus holds on a topic, or
    checking why a question was refused. For answering a user's question,
    prefer answer_question, which will refuse when the corpus cannot support
    an answer.

    Args:
        query: Search query or keywords.
        k: How many passages to return (1-20).

    Returns:
        Ranked passages with retrieval score and section-level citation path.
    """
    query = (query or "").strip()
    if not query:
        return "ERROR: query must not be empty."

    try:
        retriever, _ = _ready()
        hits = retriever.search(query, k=_clamp(k, 5))
    except Exception as exc:
        log.exception("search_documents failed")
        return f"ERROR: {exc}"

    if not hits:
        return "No passage in the corpus matched this query."

    blocks = [
        f"[{i}] {h.chunk.doc_title} — {h.chunk.section_path}\n"
        f"score: {h.score:.2f} | doc_id: {h.chunk.doc_id} | "
        f"chunk_id: {h.chunk.chunk_id}\n\n{h.chunk.text}"
        for i, h in enumerate(hits, start=1)
    ]
    log.info("search q=%r -> %d hits", query, len(hits))
    return "\n\n---\n\n".join(blocks)


@mcp.tool()
def list_documents() -> str:
    """List the indexed documents, with chunk counts and the active backend.

    Useful for telling the user what this corpus does and does not cover,
    especially after a refusal.
    """
    try:
        retriever, manifest = _ready()
    except Exception as exc:
        log.exception("list_documents failed")
        return f"ERROR: {exc}"

    lines = [
        f"Backend: {retriever.name}",
        f"Corpus:  {_corpus_dir()}",
        f"Indexed: {manifest.generated_at}",
        "",
    ]
    for d in manifest.docs:
        row = f"- {d.title}  ({d.doc_id}, {d.chunk_count} chunks)"
        if getattr(d, "error", ""):
            row += f"  [INGEST ERROR: {d.error}]"
        lines.append(row)
    return "\n".join(lines)


if __name__ == "__main__":
    if TRANSPORT == "stdio":
        mcp.run(transport="stdio")
    else:
        # Build the index before binding the port. On http the first request
        # usually arrives from a client that has already timed out its
        # handshake, so paying the embedding cost up front is the difference
        # between "slow first call" and "server looks broken".
        _ready()

        if not TOKEN:
            # Refusing to start is the right call. Behind a tunnel this
            # process is on the public internet, and a forgotten env var
            # would otherwise publish the whole corpus silently. A random
            # subdomain is not a secret: Cloudflare's certificates land in
            # public Certificate Transparency logs within minutes.
            log.error("MCP_TOKEN is not set; refusing to serve http")
            sys.exit(1)

        import uvicorn

        # Deliberately a raw ASGI middleware and not Starlette's
        # BaseHTTPMiddleware. The latter buffers the response body to hand it
        # to your dispatch function, which quietly breaks streaming: the
        # initialize call still returns 200, then the SSE stream never
        # arrives and the client reports the server as unreachable. At this
        # layer the response is passed through untouched.
        OAUTH_PREFIX = "/.well-known/"
        OAUTH_EXACT = "/register"

        async def _reject(send, status: int, message: str) -> None:
            body = f'{{"error":"{message}"}}'.encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": status,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-length", str(len(body)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": body})

        def _client_ip(headers: dict[bytes, bytes], scope) -> str:
            """The real caller, not the last proxy.

            Behind a tunnel scope["client"] is always a Cloudflare edge address,
            which makes it useless for logging. CF-Connecting-IP carries the
            original caller and is set by Cloudflare itself.

            Trusting a client-supplied header is normally a mistake -- anyone
            can send X-Forwarded-For. It is safe here only because the server
            binds 127.0.0.1, so the tunnel is the sole path in and Cloudflare
            overwrites the header on every request. Expose this port directly
            and the trust assumption is gone.
            """
            for name in (b"cf-connecting-ip", b"x-forwarded-for"):
                value = headers.get(name)
                if value:
                    return value.decode("latin-1").split(",")[0].strip()
            client = scope.get("client")
            return client[0] if client else "?"

        def _describe(body: bytes) -> str:
            """Turn a JSON-RPC body into something readable.

            Every MCP call is POST /mcp; what distinguishes them lives in the
            body. Without this the log is a wall of identical lines and you
            cannot tell a handshake from a question.
            """
            if not body:
                return "-"
            try:
                msg = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                return "unparseable"
            if isinstance(msg, list):
                return f"batch[{len(msg)}]"
            method = msg.get("method") or ("response" if "result" in msg else "?")
            if method == "tools/call":
                name = (msg.get("params") or {}).get("name")
                if name:
                    return f"tools/call {name}"
            return method

        class BearerAuth:
            def __init__(self, app):
                self.app = app

            async def __call__(self, scope, receive, send):
                if scope["type"] != "http":
                    return await self.app(scope, receive, send)

                path = scope.get("path", "")
                headers = dict(scope.get("headers", []))
                ip = _client_ip(headers, scope)
                method = scope.get("method", "?")

                # A 401 on the OAuth discovery paths tells a client "this
                # server speaks OAuth, go negotiate" and it will loop, because
                # there is no OAuth here. 404 reads as "no OAuth support",
                # after which the client is content with the token in the URL.
                if path.startswith(OAUTH_PREFIX) or path == OAUTH_EXACT:
                    log.info("%-15s %-8s %-26s -> 404 no oauth", ip, "-", path)
                    return await _reject(send, 404, "not found")

                credential = ""
                source = ""
                auth = headers.get(b"authorization")
                if auth:
                    scheme, _, rest = auth.decode("latin-1").partition(" ")
                    if scheme.lower() == "bearer" and rest:
                        credential, source = rest, "header"

                if not credential:
                    # Claude Desktop's custom connector form accepts a URL and
                    # nothing else -- no header field, no OAuth for a server
                    # this small. So the token is also read from ?token=...
                    # Strictly worse than a header, since URLs reach proxy
                    # logs, but it is the difference between "works" and
                    # "cannot be connected".
                    qs = parse_qs(scope.get("query_string", b"").decode("latin-1"))
                    credential = (qs.get("token") or [""])[0]
                    if credential:
                        source = "query"

                # The log records whether a token was presented and whether it
                # was right -- never the token itself. Behind a public tunnel
                # this file is the only record of who tried what, and a log you
                # cannot screen-share is a log you will not read.
                if not credential:
                    status = "<no-token>"
                elif secrets.compare_digest(credential, TOKEN):
                    status = f"<valid-token:{source}>"
                else:
                    status = f"<invalid-token:{source}>"

                session = headers.get(b"mcp-session-id", b"").decode("latin-1")
                session = session[:6] if session else "-"

                # compare_digest, not ==, so the comparison time does not
                # depend on how many leading characters were guessed right.
                if not secrets.compare_digest(credential, TOKEN):
                    log.warning(
                        "%-15s %-8s %-26s -> 401 %s", ip, session, f"{method} {path}", status
                    )
                    return await _reject(send, 401, "unauthorized")

                if method != "POST":
                    # GET opens the server-to-client event stream; DELETE ends
                    # the session. Neither carries a JSON-RPC body.
                    what = "open event stream" if method == "GET" else "close session"
                    log.info("%-15s %-8s %-26s %s", ip, session, what, status)
                    return await self.app(scope, receive, send)

                # Read the body to see what this call actually is, then hand
                # the untouched messages back to the app. ASGI receive() can
                # only be consumed once, so anything read here has to be
                # replayed or the request arrives empty.
                buffered: list[dict] = []
                body = b""
                while True:
                    message = await receive()
                    buffered.append(message)
                    if message["type"] != "http.request":
                        break
                    if len(body) < 65536:
                        body += message.get("body", b"")
                    if not message.get("more_body", False):
                        break

                async def replay():
                    if buffered:
                        return buffered.pop(0)
                    return await receive()

                log.info("%-15s %-8s %-26s %s", ip, session, _describe(body), status)
                await self.app(scope, replay, send)

        app = BearerAuth(mcp.streamable_http_app())

        log.info("listening on http://%s:%d/mcp (bearer auth on)", HOST, PORT)
        # access_log=False on purpose. Uvicorn logs the full request line,
        # query string included -- which means the token, in plaintext, on
        # every single request. The middleware above logs the same information
        # without it.
        uvicorn.run(app, host=HOST, port=PORT, log_level="info", access_log=False)
