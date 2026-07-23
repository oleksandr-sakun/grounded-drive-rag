# Backlog

## Portfolio gaps (these show up in almost every AI job posting)
- [ ] LangGraph: rewrite the retrieval pipeline as a graph, gates as nodes.
      answer_question decomposes into this naturally.
- [ ] pgvector: dense backend on Postgres instead of the JSON embedding cache.
      Gives an honest "hybrid: BM25 + pgvector" instead of a file.
- [x] MCP server (done 2026-07-23)

## drivebot
- [ ] GitHub repo description says nothing about MCP — the newest and
      strongest part of the project.
- [ ] OAuth 2.1 + Dynamic Client Registration. Required by ChatGPT, accepted
      by Claude Desktop, and the line between a demo and something a company
      can deploy. SDK provides the routes; ~9 provider methods to implement.
      Needs the stable domain, which now exists.
- [ ] dev-serve.sh is redundant now that systemd runs both services. Keep for
      local work without systemd, or delete.
- [ ] build_context duplicates section_path: chunk.text already starts with it.
      Wastes tokens on every request, in the Slack bot too. Re-run evals after.

## Infrastructure (matrix)
- [ ] /opt/freelance/ is not in any backup. secrets/tunnel-credentials.json
      and .env exist in one copy only. Add /opt/backups/drivebot/ following
      the pattern of /opt/floa/deploy/backup.sh.
- [ ] Offsite backup still missing. /opt/backups sits on the same sda as
      everything it protects — floa, sage, sheriff included.

## osakun.dev
- [ ] Landing page on the root domain. Without it, anyone who trims
      drivebot.osakun.dev back to the root finds nothing, which is worse than
      a random tunnel hostname. Name, one line of positioning, links to the
      repos, the demo video. Cloudflare Pages, free.
