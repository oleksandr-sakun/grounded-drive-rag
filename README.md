# Grounded Drive Assistant

A question-answering bot over a Google Drive folder that answers **only** from
the documents, cites what it used, and says "I don't know" when it doesn't.

That last part is the whole product. Retrieval is easy. Refusal is hard.

---

## Run it

Retrieval demo needs no API key and no cloud account:

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd src
python3 cli.py index                 # build the index, print the manifest
python3 cli.py eval --no-llm         # run the test set
python3 cli.py ask "how many vacation days do I get?" --no-llm
```

For generated answers, copy `.env.example` to `.env` and set `GEMINI_API_KEY`
(or `MODEL_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`), then drop `--no-llm`.

Against a real Drive folder:

```
export GOOGLE_APPLICATION_CREDENTIALS=./sa-key.json
python3 drive_ingest.py --folder-id <folder-id> --out ../corpus-drive
python3 cli.py --corpus ../corpus-drive eval --no-llm
```

Retrieval backend is one environment variable:

```
RAG_BACKEND=bm25     # keyword. no network, no bill, no latency
RAG_BACKEND=hybrid   # BM25 + Gemini embeddings, rank-fused
RAG_BACKEND=vertex   # Vertex AI Search (Discovery Engine)
```

---

## What it looks like

**Answering, with a citation that links back to the source document:**

![Slack answer](docs/slack-answer.webp)

**Refusing, because the corpus has nothing on sabbaticals:**

![Slack refusal](docs/slack-refusal.webp)

The second screenshot is the product. The first one every RAG demo can do.

---

## The eval set is the point

```
$ python3 cli.py eval --no-llm
backend: local-bm25   corpus: corpus  (26 chunks)

ANSWERABLE   — these must NOT be refused     10/10
UNANSWERABLE — these MUST be refused           6/6
16/16 passed

$ python3 cli.py --corpus ../corpus-drive eval --no-llm
backend: local-bm25   corpus: corpus-drive  (24 chunks)
16/16 passed
```

Any RAG demo can answer questions. This one is tested on questions it must
*refuse*: crypto payment policy, parental leave, who the CEO is, whether you can
bring a dog to the office. None of that exists in the corpus, so any answer would
be an invention.

The same 16 cases pass against two different corpora: hand-written `.md` files,
and a Google Drive export where the security policy arrives as a signed PDF and
pricing lives in a Sheet. Different files, different headings, different scores —
same verdicts. The gate is anchored to the documents, not tuned to one set of them.

A RAG system that quietly guesses is worse than no system, because a wrong answer
delivered confidently — with a citation attached — is harder to catch than no
answer at all.

---

## Seven things that broke

Everything below was hit while building this. Each one is reproducible from this
repository.

### 1. A score threshold is not a grounding gate

The obvious approach: refuse if the retrieval score is below X. It failed in both
directions on the first run — and it fails in both directions on the numbers this
repository still produces today:

| Question | BM25 score | Score threshold | Concept coverage |
| --- | --- | --- | --- |
| `what is a KX-Code?` | 1.76 | **refused** — the answer was right there in the glossary | **answered**, cites Glossary |
| `what's our policy on crypto payments?` | 10.08 | **answered** — confidently retrieved the *vacation policy*, because the word "policy" appears in every document | **refused** |

The question the corpus cannot answer scores six times higher than one it can.
No threshold exists that gets both of these right.

BM25 scores are not comparable across queries. They scale with query length and
with how rare the chosen words happen to be. Thresholding an incomparable number
is superstition.

**What replaced it: concept coverage.** Take the most distinctive idea in the
question and ask one thing that does have a stable answer:

> does that idea appear, in any of its forms, anywhere in the text we retrieved?

If the user asks about crypto and "crypto" appears nowhere in the corpus, no
similarity score gets to talk us into answering. If they ask about a KX-Code and
it's sitting in the glossary, no low score gets to talk us out of it.

### 2. The new gate then refused things it shouldn't have

First run of the coverage gate: `how many vacation days do I get?` → **refused**.

The word "many" appears nowhere in the corpus, so the gate concluded the corpus
had never heard of the topic. It had. It just doesn't say "many".

Same for `get`. Same for `isn` — which is what `isn't` becomes if you tokenise
carelessly.

**Fix:** contraction handling before tokenisation, a thorough function-word
stoplist, and light stemming so `carry` finds `carried` and `payments` isn't a
different word from `payment`. The stemmer is crude and doesn't need to be
otherwise — it needs to be *consistent*, applied identically at index time and at
query time.

That is the actual work in a RAG system. The pipeline is an afternoon.

### 3. Semantic search cannot say "I don't know"

The most important finding here, and the reason the gate is built the way it is.

Switching on the hybrid backend, the six unanswerable questions scored:

```
15.75  what's our policy on crypto payments?
16.26  how do I request a sabbatical?
16.13  what's the parental leave allowance?
16.13  what's the company's revenue?
16.26  can I bring a dog to the office?
```

Near the top of the range. The embedding model was *confident* about questions the
corpus cannot answer.

This is not a bug. It is what embeddings are: a nearest-neighbour search always
returns a nearest neighbour. There is no vector for "nothing here is relevant".
BM25 is at least honest — no matching words means a low score. Embeddings always
look sure of themselves.

**So the hallucination risk goes UP when you add semantics, not down.**

The gate survives this only because it doesn't ask the index. It asks the
*corpus*: is the word `sabbatical` present in any document at all? No → stop,
before the model is ever called. Grounding is anchored to the documents, not to
whichever retriever happens to be switched on.

If you take one idea from this repository, take that one.

### 4. Scale does not fix hallucination — it hides it

The intuition that more documents means fewer wrong answers is backwards.

- **More plausible noise.** In 8 documents there is obviously nothing about
  sabbaticals. In 800 there is something about leave policy, unpaid absence,
  career breaks — and the nearest neighbour becomes far more convincing. The
  model receives context that is *almost* on topic and starts bridging the gap.
- **The gaps become invisible.** With 8 documents you know what isn't there. With
  800, nobody does — not you, and not the client.

What changes with scale isn't answer quality. It's the cost of being wrong. In a
small corpus a bad answer gets noticed. In a large one, a confident answer with a
citation attached to something adjacent does not.

Refusal is therefore a first-day architectural decision, not a later hardening
step.

### 5. Vertex AI Search's Drive connector requires a Google Workspace organisation

Attempting to create a Drive-connected data store on a personal Google account
dead-ends:

![Vertex AI Search requires a Workspace organisation](docs/vertex-requires-org.webp)

> **Identity provider not configurable.** Your project must belong to an
> organization to configure identity providers.

The chain: the Drive connector mandates ACL enforcement → ACL enforcement
requires Google Identity → Google Identity requires a Workspace organisation.
None of these steps is optional.

So the managed connector — the thing every tutorial reaches for — is unavailable
to anyone without a corporate Workspace. That is worth establishing *before*
promising a client an architecture, not after.

This is also why `drive_ingest.py` exists, and why it stopped being the fallback
and became the primary path.

### 6. The connector asks for far more permission than it uses

The Drive connector's default OAuth consent screen requests, among others:

> View, edit, create and delete all your files in Google Drive.

For a system that only ever reads. Unticking the write scopes and granting only
read — authentication still succeeds. The connector asked for delete permission
it does not need, and the default is *select all*.

`drive_ingest.py` uses `drive.readonly` on a single shared folder.

### 7. In the connector, "just this folder" is config, not permission

The connector lets you restrict indexing to specific folder IDs. But the OAuth
token it holds still has access to the entire Drive — the folder restriction is a
filter in the connector's configuration, not a boundary on the credential.

Change the config, and the restriction is gone. The access was always there.

With a service account shared into one folder, the boundary is on the credential
itself. That is the difference between *does not read* and *cannot read*.

---

## Architecture

```
Drive folder
   │  Drive API — service account, drive.readonly, one shared folder
   ▼
Docs   → Markdown (native export)
Sheets → Markdown tables (structure preserved: a number
         without its column header is worthless to retrieval)
PDF    → Markdown (PyMuPDF, reading-order blocks)
   │
   ▼
corpus/*.md  +  manifest.json
   │
   ▼
chunk on headings — not a sliding window. Headings give stable
                    citation anchors and match how policy documents
                    are actually written.
   │
   ▼
Retriever  (RAG_BACKEND)
   ├── bm25     keyword + hand-maintained term dictionary
   ├── hybrid   BM25 + Gemini embeddings, fused on RANK not score
   └── vertex   Vertex AI Search (Discovery Engine)
   │
   ▼
Gate 1: concept coverage  ← deterministic, corpus-anchored, refuses
                            before a single token is spent
Gate 2: grounded prompt   ← cites sources, returns NO_ANSWER when
                            the retrieved text does not contain the answer
   │
   ▼
Answer + citations → link back to the source document
```

### Why RRF and not a weighted sum

BM25 scores and cosine similarities are different currencies — one is unbounded
and query-dependent, the other is bounded to [-1, 1]. Adding them together
produces a number, but not a meaning.

Reciprocal Rank Fusion only asks each retriever *how highly did you rank this*,
which is a question both can answer honestly.

### Why the manifest exists

Every source file gets a row: id, revision, content hash, chunk count, and any
error. Reconciliation becomes a set comparison, so a missing document is a loud,
specific failure rather than a vague sense that the bot has got worse.

The manifest also tracks an `empty` list: documents that parsed without error and
produced zero chunks. That is the signature of a scanned PDF — present in every
summary, containing nothing. Present-but-empty is worse than failed, because
failure at least announces itself.

An index you cannot audit is an index you cannot trust.

### Why the retriever sits behind an interface

One environment variable swaps the backend. Nothing above the interface knows or
cares.

That is what makes it possible to move between a self-hosted index and a managed
one — in either direction — without rewriting the application. A client should not
have to choose between "the right default" and "not locked in".

---

## Interface: Slack first, not Slack only

People don't visit a separate website to ask a question. They ask where they
already are.

A standalone web chat gets used in week one, forgotten by week three, and quietly
cancelled at renewal. Put the assistant where the work already happens and it
costs nothing to reach. So Slack is the default here — not the limit.

The core knows nothing about Slack. `retrieval.py`, `answering.py`, `chunking.py`
and `manifest.py` contain no transport code at all, and they are already driven by
two independent front ends in this repository:

```
corpus → retrieval → gate → answering + citations
                      ▲
       ┌──────────────┼──────────────┐
    cli.py      slack_bot.py   mcp_server.py    web API
                                                (not built)
```

That is why a third front end is an adapter, not a rewrite. A React/Next.js chat
with token streaming and clickable citations is a thin layer over the same call
the CLI already makes — worth building when the users live in a web product, or
sit outside the workspace entirely; not worth building *first*, when they don't.

The interface is a deployment decision. The grounding is an architectural one.
Only one of them is expensive to get wrong.

---

## Interface: MCP

Slack needed `slack_bot.py`. Cursor would need another file. A customer's own
chat would need a third. Each one is a week of someone's life spent on transport
code that teaches nothing.

MCP collapses that into one server. `mcp_server.py` exposes the same retrieval
the Slack bot uses — `answer_question`, `search_documents`, `list_documents` —
and any MCP client connects to it without a line of new code. Claude Desktop,
Cursor, Claude Code, a phone. The core still knows nothing about any of them.

```
corpus → retrieval → gate → answering + citations
                      ▲
       ┌──────────────┼──────────────┬──────────────┐
    cli.py      slack_bot.py   mcp_server.py    web API
                                                (not built)
```

**`answer_question` does not call an LLM, on purpose.** The MCP client is
already a model; a second one here would add a round trip, a bill, and a second
place for the answer to drift. So the server runs Gate 1 — the deterministic
half, the one that actually prevents hallucination — and returns either numbered
sources with citation rules, or a refusal naming the gate that fired. The client
model becomes Gate 2.

The consequence is the part worth showing. Ask an MCP client something the
corpus has never heard of and it does not invent a plausible answer:

```
REFUSED (gate: unknown-term)
```

The grounding gate survives the protocol boundary. That is the whole claim of
this repository, demonstrated in someone else's chat window rather than mine.

Two transports. `stdio` by default, for a client that spawns the process
directly. `streamable-http` behind a tunnel for everything else — a phone, a
sandboxed desktop app, a client on another continent. Same tools, same gate.

```bash
./dev-serve.sh                                    # tunnel + http server, prints the URL
cd src && ../.venv/bin/python smoke_mcp.py        # stdio, 5/5
cd src && ../.venv/bin/python smoke_http.py URL   # http, 5/5
```

Auth on http is a bearer token, accepted in the `Authorization` header or as
`?token=`. The query parameter is strictly worse — URLs reach proxy logs and
browser history — but a client whose connector form accepts a URL and nothing
else leaves no other option. The server refuses to start on http without a
token: behind a tunnel this process is on the public internet, and a forgotten
environment variable would otherwise publish the corpus in silence.

---

## Five more things that broke: remote MCP

Same rule as before — everything here was hit while building it, and every one
of them presents as a different problem than it is.

### 8. stdio over ssh dies before it can tell you why

The obvious way to reach a server on another machine: `command: ssh`, and let
the client talk to the remote process over stdin/stdout. It fails, and the log
says only this:

```
Server started and connected successfully
Server transport closed          ← 96 ms later
```

96 ms is not enough time to open a TCP connection, let alone start Python. The
process died before the network. `ssh -vvv -E logfile` produced no file at all,
which narrows it further: the binary exits before it initialises logging.

The same command pasted into a terminal works and hangs waiting for stdin,
exactly as it should. So the command is right and the environment it runs in is
wrong — a packaged desktop application does not hand its children the
environment your shell has, and `~` is not where you think it is.

**What replaced it:** http. Which turned out to be the better answer anyway —
see below.

### 9. `421 Misdirected Request` comes from the SDK, not the proxy

The tunnel worked. `curl` to the public URL returned `421`, which reads like a
Cloudflare problem and is not one.

The MCP SDK has DNS-rebinding protection: it checks the `Host` header against an
allowlist and answers `421` to anything unlisted. Localhost is allowed by
default. Behind a tunnel the Host becomes the public hostname, which is not.

```python
TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[PUBLIC_HOST, "127.0.0.1:*", "localhost:*"],
)
```

The check is correct and worth keeping — `Host: evil.com` still gets `421`. It
just has to be told which hostname is legitimately yours.

### 10. A `401` on `/.well-known` starts an OAuth loop that never ends

With the token working, the handshake returned `200` and the client still
reported the server as unreachable. The log explains it:

```
POST /mcp?token=...                              200 OK
GET  /.well-known/oauth-protected-resource/mcp   401
GET  /.well-known/oauth-authorization-server     401
POST /register                                   401
POST /mcp?token=...                              200 OK      ← starts over
```

The auth middleware was answering `401` to *everything*, including the OAuth
discovery paths, and attaching `WWW-Authenticate: Bearer`. To a client that is
not "wrong token" — it is "this server speaks OAuth, go negotiate". So it
negotiates, gets `401` again, and loops.

**The fix is counter-intuitive: return `404`, not `401`.** A `404` on those
paths means "no OAuth here", after which the client is content with the token it
already has. The `WWW-Authenticate` header goes too — that header *is* the
signal that starts the dance.

A client that connected while the server still behaved the old way caches the
conclusion. Deleting and re-adding the connector was required; restarting was
not enough.

### 11. `BaseHTTPMiddleware` silently breaks the stream

The natural way to add auth to a Starlette app:

```python
class BearerAuth(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next): ...
```

`initialize` returns `200`. Nothing after it works. The client reports the
server as unreachable; the server log shows one successful request and then
silence — no error on either side.

`BaseHTTPMiddleware` buffers the response body in order to hand it to
`dispatch`. Streamable HTTP is an SSE stream. Buffering a stream that never ends
means it never arrives.

**What replaced it:** a raw ASGI middleware, which inspects `scope` and passes
`receive`/`send` through untouched. Roughly the same number of lines, and the
response is never materialised.

This is also why `smoke_http.py` exercises `tools/list` and real tool calls
rather than just the handshake. A test that stops at `initialize` passes on a
server that is completely broken.

### 12. `cloudflared` inherits a config file you didn't mean it to

A quick tunnel came up, registered a subdomain, and returned `404` to every
path — including paths that should have returned `401` from our own middleware.
Nothing appeared in the tunnel's log, meaning requests were never reaching the
machine.

The cause was in the startup output all along:

```
Settings: map[cred-file:/home/user/.cloudflared/0c3ff864-....json ...]
```

A quick tunnel has no credentials. It had picked up `~/.cloudflared/config.yml`
belonging to a *different*, named tunnel running on the same host, and routed by
those rules instead.

```bash
cloudflared --config /dev/null tunnel --url http://127.0.0.1:8765
```

`--config /dev/null` is load-bearing on any host that already runs a named
tunnel. `dev-serve.sh` passes it.

### Also worth knowing

**A remote MCP URL must be `https`.** Which rules out reaching a server across
your own LAN by IP, however local the setup — the tunnel is not optional even
when the two machines are a metre apart.

**Quick tunnels get a new subdomain on every start.** The connector URL has to
be updated each time. `dev-serve.sh` at least prints the new one, complete with
token, ready to paste. A named tunnel fixes it properly and needs a domain.

---

## Security

- Read-only Drive scope, scoped to one shared folder. No broader Drive access.
- Documents are not used for training. They enter a prompt for the duration of one
  request and are gone.
- Query logs kept separate from document content.
- No document text is written anywhere except the corpus directory and the index.

---

## Known limits

An honest list, because a demo that hides its edges is a sales pitch.

- **Scanned PDFs need OCR.** Not wired up. The manifest flags them as empty rather
  than pretending they are fine.
- **The term dictionary is hand-maintained.** In production it grows every time the
  bot gets a question wrong — the cheapest possible feedback loop, but a loop, not
  a solve. The hybrid backend reduces the dependence on it; it does not remove it.
- **RRF scores are not diagnostic.** Any chunk ranked first by both retrievers gets
  the same fused score, so the number in the logs stops being informative. Real,
  and cosmetic.
- **No incremental sync yet.** `drive_ingest.py` re-reads the whole folder. The
  manifest already carries revision and content hash, so skipping unchanged files
  is a small change — it just isn't done.
- **The corpus here is fictional.** Kestrel Labs does not exist. The gaps in it are
  deliberate: they are what makes the refusal tests meaningful.
