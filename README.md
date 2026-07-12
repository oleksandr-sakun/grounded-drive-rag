# Grounded Drive Assistant

A question-answering bot over a Google Drive folder that **only** answers from
the documents, cites what it used, and says "I don't know" when it doesn't.

That last part is the whole product. Retrieval is easy. Refusal is hard.

---

## Run it

No API key needed for the retrieval demo:

```bash
cd src
python3 cli.py index                 # build the index, print the manifest
python3 cli.py eval --no-llm         # run the test set
python3 cli.py ask "how many vacation days do I get?" --no-llm
```

For full answers, set `ANTHROPIC_API_KEY` (or `MODEL_PROVIDER=gemini` +
`GEMINI_API_KEY`) and drop `--no-llm`.

Against a real Drive folder:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=./sa-key.json
python3 drive_ingest.py --folder-id <folder-id> --out ../corpus
```

---

## The eval set is the point

```
ANSWERABLE — these must NOT be refused          10/10
UNANSWERABLE — these MUST be refused             6/6
```

Every RAG demo can answer questions. This one is tested on questions it must
*refuse*: crypto payment policy, parental leave, who the CEO is. None of that
exists in the corpus, so any answer would be an invention.

A RAG system that quietly guesses is worse than no system at all, because a
wrong answer delivered confidently — with a citation attached — is harder to
catch than no answer.

---

## Two things that failed, and what replaced them

### 1. The score threshold didn't work

The obvious grounding gate is "refuse if the retrieval score is below X". It
failed in both directions on the first run:

| Question | Score | What happened |
|---|---|---|
| `what is a KX-Code?` | 1.83 | **Refused** — but the answer was right there in the glossary |
| `what's our policy on crypto payments?` | 9.60 | **Answered** — confidently retrieved the *vacation policy*, because the word "policy" appears in every document |

BM25 scores are not comparable across queries. They scale with query length and
with how rare the words happen to be. Thresholding an incomparable number is
superstition.

**What replaced it: concept coverage.** Take the most distinctive idea in the
question. Ask one question that actually has a stable answer:

> does that idea appear, in any of its forms, in the text we retrieved?

If the user asks about crypto and "crypto" appears nowhere in the corpus, no
similarity score gets to talk us into answering. If they ask about a KX-Code and
it's sitting in the glossary, no low score gets to talk us out of it.

Cheap, deterministic, and it explains itself in the logs.

### 2. The gate then refused things it shouldn't have

First run of the new gate: `how many vacation days do I get?` → **refused**.

The word "many" appears nowhere in the corpus, so the gate concluded the corpus
had never heard of the topic. It had. It just doesn't say "many".

Same for `get`, and for `isn` — which is what `isn't` becomes if you tokenise
carelessly.

**Fix:** contraction handling before tokenisation, a thorough function-word
stoplist, and light stemming so `carry` finds `carried` and `payments` is not a
different word from `payment`.

The stemmer is crude and doesn't need to be otherwise. It needs to be
*consistent* — applied identically at index time and query time.

That's the actual work in a RAG system. Not the pipeline. The pipeline is an
afternoon.

---

## Architecture

```
Drive folder
   │  Drive API (read-only, service account)
   ▼
Docs → Markdown (native export)
Sheets → Markdown tables (structure preserved — a number
         without its column header is worthless to retrieval)
PDF → Markdown (PyMuPDF, reading-order blocks)
   │
   ▼
corpus/*.md  +  manifest.json
   │
   ▼
chunk on headings (not a sliding window — headings give
                   stable citation anchors and match how
                   policy documents are actually written)
   │
   ▼
Retriever (pluggable)
   ├── LocalBM25          self-hosted, no bill, no latency
   └── VertexAISearch     managed (Discovery Engine)
   │
   ▼
Gate 1: concept coverage   ← deterministic, free, refuses before spending a token
Gate 2: grounded prompt    ← cites sources, returns NO_ANSWER when insufficient
   │
   ▼
Answer + citations → back to the source document
```

### Why the manifest exists

Every source file gets a row: id, revision, content hash, chunk count, and any
error. Reconciliation is a set comparison, so a missing document is a loud,
specific failure instead of a vague sense that the bot got worse.

This is not theoretical. On a previous migration a managed connector reported a
successful sync and indexed **151 of an expected 1,035 documents**. No error. No
warning. The bot simply didn't know things, and there was no way to see it from
the outside.

The manifest also has an `empty` list — documents that parsed without error and
produced zero chunks. That is the signature of a scanned PDF: present in every
summary, containing nothing. Present-but-empty is worse than failed, because
failed at least tells you.

### Why the retriever is behind an interface

One config value swaps `local` for `vertex`. Nothing above the interface knows
or cares. That's what makes it possible to move between a self-hosted index and
a managed one — in either direction — without rewriting the application.

Vertex AI Search is the right default for a Drive folder: Google operates the
connector, the parsing and the sync. But "right default" and "locked in" are
different things, and a client should not have to pick one to get the other.

### Two things worth knowing about Vertex AI Search

Both cost real time to discover:

1. **Regional endpoints are real.** A data store in `eu` must be queried at
   `eu-discoveryengine.googleapis.com`. The global host returns an empty result
   set — not an error.

2. **ACL enforcement is immutable.** If the data store was created by a connector
   with `aclEnabled: true`, a service account will authenticate successfully and
   receive *zero results*. No error, no warning. You need an OAuth identity that
   actually holds access to the underlying documents.

   A Drive folder you own and share with a service account does not have this
   problem. A corporate folder wired through SSO very well might.

---

## Interface: Slack, not a web app

People don't visit a separate website to ask a question. They ask where they
already are.

A standalone web chat is the version that gets used in week one, forgotten by
week three, and quietly cancelled at renewal. Put the bot where the work already
happens and it costs nothing to reach.

A web widget is a reasonable phase two, if people outside the workspace need
access. It is not phase one.

---

## Security

- Read-only Drive scope, scoped to one folder. No broader Drive access.
- Documents are not used for training.
- Query logs are kept separate from document content.
- No document text is written anywhere except the corpus directory and the index.

---

## Known limits

Honest list, because a demo that hides its edges is a sales pitch:

- **Scanned PDFs need OCR.** Not wired up. The manifest will flag them as empty
  rather than pretend they're fine.
- **BM25 has no semantics.** "Time off" finds "vacation" only because the term
  dictionary says so. That dictionary is hand-maintained, and in production it
  grows every time the bot gets a question wrong — which is the cheapest possible
  feedback loop, but it is a loop, not a solve. The Vertex backend covers this
  properly.
- **The stemmer is crude.** Consistent, but crude.
- **No incremental sync yet.** `drive_ingest.py` re-reads the whole folder. The
  manifest already carries revision and content hash, so skipping unchanged
  files is a small change — it just isn't done.
