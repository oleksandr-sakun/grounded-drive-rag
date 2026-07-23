#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${MCP_PORT:-8765}"
BACKEND="${RAG_BACKEND:-hybrid}"
CORPUS="${CORPUS_DIR:-corpus}"
TUNNEL_LOG="$(mktemp -t drivebot-tunnel.XXXXXX.log)"

TOKEN="$(grep -E '^MCP_TOKEN=' "$ROOT/.env" 2>/dev/null | cut -d= -f2- || true)"
if [[ -z "$TOKEN" ]]; then
  echo "error: MCP_TOKEN is missing from $ROOT/.env" >&2
  exit 1
fi

cleanup() {
  [[ -n "${TUNNEL_PID:-}" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
  rm -f "$TUNNEL_LOG"
}
trap cleanup EXIT INT TERM

echo "starting tunnel..."
cloudflared --config /dev/null tunnel --url "http://127.0.0.1:$PORT" >"$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!

PUB=""
for _ in $(seq 1 60); do
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "error: cloudflared exited" >&2; tail -20 "$TUNNEL_LOG" >&2; exit 1
  fi
  PUB="$(grep -oE '[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)"
  [[ -n "$PUB" ]] && break
  sleep 1
done

if [[ -z "$PUB" ]]; then
  echo "error: no tunnel URL after 60s" >&2; tail -20 "$TUNNEL_LOG" >&2; exit 1
fi

printf '\n-------------------------------------------------------------\n'
printf 'connector URL:\n\n  https://%s/mcp?token=%s\n' "$PUB" "$TOKEN"
printf '\nbackend: %s   corpus: %s   port: %s\n' "$BACKEND" "$CORPUS" "$PORT"
printf -- '-------------------------------------------------------------\n\n'

cd "$ROOT/src"
exec env \
  MCP_TRANSPORT=streamable-http \
  MCP_PUBLIC_HOST="$PUB" \
  MCP_PORT="$PORT" \
  RAG_BACKEND="$BACKEND" \
  CORPUS_DIR="$CORPUS" \
  "$ROOT/.venv/bin/python" mcp_server.py
