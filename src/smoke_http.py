#!/usr/bin/env python3
"""
End-to-end check of the http transport.

Unlike a curl of `initialize`, this runs the whole session the way a real
client does: handshake, tools/list, then actual tool calls. That distinction
matters -- a broken middleware can pass the handshake and still fail every
request after it, which looks like "server unreachable" on the client side and
like a single 200 OK in the server log.

    cd src && ../.venv/bin/python smoke_http.py https://host/mcp?token=...
    cd src && ../.venv/bin/python smoke_http.py            # localhost, token from .env
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

ROOT = Path(__file__).resolve().parent.parent

CASES = [
    ("answer_question", {"question": "how many vacation days do I get?"}, "sources"),
    ("answer_question", {"question": "what is our crypto payment policy?"}, "refuse"),
    ("answer_question", {"question": "what is a KX-Code?"}, "sources"),
    ("search_documents", {"query": "sev1 escalation", "k": 2}, "any"),
    ("list_documents", {}, "any"),
]


def default_url() -> str:
    token = os.environ.get("MCP_TOKEN", "")
    if not token:
        env = ROOT / ".env"
        if env.is_file():
            for line in env.read_text().splitlines():
                if line.startswith("MCP_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break
    port = os.environ.get("MCP_PORT", "8765")
    return f"http://127.0.0.1:{port}/mcp?token={token}"


async def main(url: str) -> int:
    shown = url.split("token=")[0] + "token=***"
    print(f"url:     {shown}\n")

    fails = 0
    async with streamablehttp_client(url) as (read, write, _):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"server:  {init.serverInfo.name}")

            tools = await session.list_tools()
            print(f"tools:   {', '.join(sorted(t.name for t in tools.tools))}\n")

            for tool, args, expect in CASES:
                result = await session.call_tool(tool, args)
                text = "".join(
                    b.text for b in result.content if getattr(b, "type", "") == "text"
                )
                refused = text.startswith("REFUSED")
                errored = text.startswith("ERROR")

                if errored:
                    ok = False
                elif expect == "refuse":
                    ok = refused
                elif expect == "sources":
                    ok = not refused
                else:
                    ok = True

                fails += 0 if ok else 1
                label = args.get("question") or args.get("query") or "-"
                verdict = "REFUSED" if refused else ("ERROR" if errored else "OK")
                print(f"  [{'PASS' if ok else 'FAIL'}] {tool:<17} {verdict:<8} {label}")
                if not ok:
                    print(f"          {text[:200]}")

    print()
    print(f"{len(CASES) - fails}/{len(CASES)} passed")
    return 1 if fails else 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else default_url()
    sys.exit(asyncio.run(main(target)))
