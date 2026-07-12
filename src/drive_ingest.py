#!/usr/bin/env python3
"""
Google Drive -> Markdown -> manifest.

This is the piece that replaces a managed connector, and the reason to write it
yourself is visibility. A connector is a black box that reports success; this
reports *what it did to every single file*. When the client asks "is everything
indexed?", the answer is a file, not a shrug.

    python3 drive_ingest.py --folder-id <id> --out ../corpus

Auth: a service account with read-only Drive scope, shared into the folder.
Set GOOGLE_APPLICATION_CREDENTIALS to the JSON key path.

    pip install google-api-python-client google-auth pymupdf

Note on ACLs: a folder you own and share with the service account works fine.
A corporate folder behind SSO may not — the service account authenticates, and
then quietly sees nothing. If document_count comes back zero, that is where to
look first, and no amount of retrying will help.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import sys
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunking import chunk_markdown  # noqa: E402
from manifest import DocEntry, Manifest, sha256  # noqa: E402

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

GOOGLE_DOC = "application/vnd.google-apps.document"
GOOGLE_SHEET = "application/vnd.google-apps.spreadsheet"
PDF = "application/pdf"

SUPPORTED = {GOOGLE_DOC, GOOGLE_SHEET, PDF, "text/markdown", "text/plain"}


def slugify(name: str) -> str:
    # Strip the extension first, or "08-glossary.md" becomes "08-glossarymd".
    name = re.sub(r"\.(md|pdf|txt|csv|docx?|xlsx?)$", "", name, flags=re.I)
    s = re.sub(r"[^\w\s-]", "", name.lower()).strip()
    return re.sub(r"[\s_]+", "-", s)[:60] or "untitled"


def drive_client():
    creds = service_account.Credentials.from_service_account_file(
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"], scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_folder(svc, folder_id: str) -> list[dict]:
    """Every file in the folder, recursively. Trashed files excluded."""
    files, stack = [], [folder_id]

    while stack:
        current = stack.pop()
        page_token = None
        while True:
            resp = (
                svc.files()
                .list(
                    q=f"'{current}' in parents and trashed = false",
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime, webViewLink)",
                    pageSize=100,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for f in resp.get("files", []):
                if f["mimeType"] == "application/vnd.google-apps.folder":
                    stack.append(f["id"])
                else:
                    files.append(f)
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    return files


# --------------------------------------------------------------------------
# Converters. Each returns Markdown, or raises.
# --------------------------------------------------------------------------


def export_doc(svc, file_id: str) -> str:
    """Google Docs export natively to Markdown. Nothing clever needed."""
    data = svc.files().export(fileId=file_id, mimeType="text/markdown").execute()
    return data.decode("utf-8") if isinstance(data, bytes) else data


def export_sheet(svc, file_id: str, title: str) -> str:
    """Sheets -> CSV -> Markdown table.

    Naive text extraction turns a spreadsheet into a wall of unlabelled values,
    and retrieval on that is worthless: a number means nothing without its
    column header. Keeping the table structure is the whole job here.
    """
    import csv

    raw = svc.files().export(fileId=file_id, mimeType="text/csv").execute()
    text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
    rows = list(csv.reader(io.StringIO(text)))
    rows = [r for r in rows if any(cell.strip() for cell in r)]

    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    header, body = rows[0], rows[1:]
    out = [f"# {title}", ""]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * width) + "|")
    for r in body:
        out.append("| " + " | ".join(c.strip() for c in r) + " |")

    return "\n".join(out)


def export_pdf(svc, file_id: str, title: str) -> str:
    """PDF -> Markdown, best effort.

    This is the only genuinely hard converter. A text-layer PDF extracts
    cleanly. A scanned one extracts *nothing*, silently — and a document that
    is present but empty is worse than one that failed, because it looks fine
    in every summary. The manifest's `empty` list exists specifically to catch
    this: parsed without error, zero chunks produced, therefore lying.
    """
    import fitz  # pymupdf

    buf = io.BytesIO()
    req = svc.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)

    doc = fitz.open(stream=buf.read(), filetype="pdf")
    parts = [f"# {title}", ""]
    for page in doc:
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: (round(b[1]), b[0]))  # reading order
        for b in blocks:
            text = b[4].strip()
            if not text:
                continue
            # Short, isolated lines at the top of a block are usually headings.
            if len(text) < 60 and "\n" not in text and not text.endswith("."):
                parts.append(f"\n## {text}\n")
            else:
                parts.append(text)
    doc.close()

    return "\n\n".join(parts)


def to_markdown(svc, f: dict) -> str:
    mime, fid, title = f["mimeType"], f["id"], f["name"]

    if mime == GOOGLE_DOC:
        return export_doc(svc, fid)
    if mime == GOOGLE_SHEET:
        return export_sheet(svc, fid, title)
    if mime == PDF:
        return export_pdf(svc, fid, title)
    if mime in ("text/markdown", "text/plain"):
        raw = svc.files().get_media(fileId=fid).execute()
        return raw.decode("utf-8") if isinstance(raw, bytes) else raw

    raise ValueError(f"unsupported mimeType: {mime}")


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder-id", required=True)
    ap.add_argument("--out", default="../corpus")
    ap.add_argument("--manifest", default="../manifest.json")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    svc = drive_client()
    files = list_folder(svc, args.folder_id)
    print(f"Drive reports {len(files)} files in the folder")

    entries: list[DocEntry] = []
    expected: set[str] = set()

    for f in files:
        fid, title, mime = f["id"], f["name"], f["mimeType"]

        if mime not in SUPPORTED:
            print(f"  skip   {title}  ({mime})")
            continue

        expected.add(fid)

        try:
            md = to_markdown(svc, f).strip()
        except Exception as exc:  # noqa: BLE001 — we want every failure recorded
            print(f"  FAIL   {title}: {exc}")
            entries.append(
                DocEntry(
                    doc_id=fid,
                    title=title,
                    source_mime=mime,
                    revision=f.get("modifiedTime", ""),
                    content_sha256="",
                    chunk_count=0,
                    web_url=f.get("webViewLink", ""),
                    error=str(exc)[:300],
                )
            )
            continue

        path = out / f"{slugify(title)}.md"
        path.write_text(md)

        chunks = chunk_markdown(md, doc_id=fid, doc_title=title)
        status = "ok  " if chunks else "EMPTY"
        print(f"  {status}   {title}  -> {path.name}  ({len(chunks)} chunks)")

        entries.append(
            DocEntry(
                doc_id=fid,
                title=title,
                source_mime=mime,
                revision=f.get("modifiedTime", ""),
                content_sha256=sha256(md),
                chunk_count=len(chunks),
                web_url=f.get("webViewLink", ""),
                local_file=path.name,
            )
        )

    man = Manifest.build(source=f"drive:{args.folder_id}", docs=entries)
    man.save(args.manifest)

    print()
    print(man.report(expected_ids=expected))

    # Fail loudly. A silent partial index is the whole problem we are solving.
    if man.failed or man.empty:
        sys.exit(1)


if __name__ == "__main__":
    main()
