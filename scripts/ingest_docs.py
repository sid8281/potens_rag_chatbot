#!/usr/bin/env python3
"""
scripts/ingest_docs.py
──────────────────────
One-shot CLI to ingest all documents in data/raw/ into ChromaDB.

Run this once before starting the FastAPI server:
    python scripts/ingest_docs.py

Options:
    --dir     Path to documents folder (default: data/raw)
    --clear   Delete all existing ChromaDB data before ingesting
    --file    Ingest a single file instead of a whole directory

Examples:
    python scripts/ingest_docs.py
    python scripts/ingest_docs.py --dir /path/to/my/docs
    python scripts/ingest_docs.py --file data/raw/gdpr.pdf
    python scripts/ingest_docs.py --clear
"""

import argparse
import sys
import time
from pathlib import Path

# ── make sure project root is on sys.path when run as a script ───────────────
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.core.ingestion import (
    delete_document,
    get_chroma_collection,
    get_collection_stats,
    ingest_directory,
    ingest_file,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def print_summary(results) -> None:
    print("\n" + "═" * 60)
    print("  INGESTION SUMMARY")
    print("═" * 60)
    for r in results:
        icon = "✓" if r.status == "success" else ("⚠" if r.status == "skipped" else "✗")
        print(f"  {icon}  {r.doc_id:<40} {r.chunks_created:>4} chunks")
        if r.error:
            print(f"       ERROR: {r.error}")
        for w in r.warnings:
            print(f"       WARN : {w}")
    print("═" * 60)
    total = sum(r.chunks_created for r in results)
    success = sum(1 for r in results if r.status == "success")
    print(f"  {success}/{len(results)} files  ·  {total} total chunks")
    print("═" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG ChromaDB vector store."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing documents to ingest (default: data/raw)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Ingest a single file instead of a directory",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Wipe the ChromaDB collection before ingesting",
    )
    args = parser.parse_args()

    print("\n" + "═" * 60)
    print("  RAG SYSTEM — Document Ingestion")
    print("═" * 60)

    # ── Optional: clear existing data ────────────────────────────────────────
    if args.clear:
        import chromadb
        from app.config import settings
        print(f"\n  Clearing collection '{settings.chroma_collection}'...")
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        try:
            client.delete_collection(settings.chroma_collection)
            print("  Collection cleared.\n")
        except Exception:
            print("  Collection did not exist — nothing to clear.\n")

    start = time.time()

    # ── Single file mode ──────────────────────────────────────────────────────
    if args.file:
        file_path = args.file.resolve()
        print(f"  Ingesting single file: {file_path}\n")
        result = ingest_file(file_path)
        print_summary([result])

    # ── Directory mode ────────────────────────────────────────────────────────
    else:
        doc_dir = args.dir.resolve()
        if not doc_dir.exists():
            print(f"\n  ERROR: Directory not found: {doc_dir}")
            print("  Create the directory and add documents, then re-run.\n")
            sys.exit(1)

        print(f"  Documents directory: {doc_dir}\n")
        results = ingest_directory(doc_dir)

        if not results:
            print(
                "\n  No documents found. Add .pdf, .txt, or .md files to:\n"
                f"  {doc_dir}\n"
            )
            sys.exit(0)

        print_summary(results)

    # ── Post-ingestion stats ──────────────────────────────────────────────────
    stats = get_collection_stats()
    elapsed = time.time() - start
    print("  ChromaDB status after ingestion:")
    print(f"  ├── Collection     : {stats['collection']}")
    print(f"  ├── Total documents: {stats['total_documents']}")
    print(f"  ├── Total chunks   : {stats['total_chunks']}")
    print(f"  ├── Documents      : {', '.join(stats['documents']) or 'none'}")
    print(f"  └── Time elapsed   : {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    main()
