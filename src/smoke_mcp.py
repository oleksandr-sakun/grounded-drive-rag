#!/usr/bin/env python3
"""
Smoke test for mcp_server.py -- no client needed.

Spawns the server over stdio exactly the way Claude Desktop would, runs the
handshake, and exercises all three tools including both edges of the grounding
gate. If this passes, the server is fine and any remaining problem is in the
client config.

    cd src && ../.venv/bin/python smoke_mcp.py
    cd src && RAG_BACKEND=hybrid ../.venv/bin/python smoke_mcp.py
"""

from __future__ import annotations

import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

CASES = [
    ("answer_question", {"question": "how many vacation days do I get?"}, "sources"),
    ("answer_question", {"question": "what is our crypto payment policy?"}, "refuse"),
    ("answer_question", {"question": "what is a KX-Code?"}, "sources"),
    ("search_documents", {"query": "sev1 escalation", "k": 2}, "any"),
    ("list_documents", {}, "any"),
]


async def main() -> int:
    params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env={
            **os.environ,
            "RAG_BACKEND": os.environ.get("RAG_BACKEND", "bm25"),
            "CORPUS_DIR": os.environ.get("CORPUS_DIR", "corpus"),
            "MCP_LOG_LEVEL": "WARNING",  # keep the smoke output readable
        },
    )

    fails = 0
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"server:  {init.serverInfo.name}")

            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"tools:   {', '.join(names)}\n")

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
    sys.exit(asyncio.run(main()))
