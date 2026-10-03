"""Offline document ingestion for the Medical RAG application.

Runs as a BATCH JOB, never inside the /chat request path.

    PDF -> PyPDFLoader -> pages -> RecursiveCharacterTextSplitter -> chunks
        -> HuggingFaceEmbeddings -> Pinecone

Usage:
    python ingestion.py                # incremental: skip files already ingested
    python ingestion.py --force        # re-ingest everything
    python ingestion.py --reset       # wipe index, then re-ingest (one-time migration)
    python ingestion.py --dry-run      # show what would change, write nothing

Design notes:
  * Chunk IDs are deterministic, so re-running replaces vectors instead of
    duplicating them.
  * A sha256 of each PDF is tracked in .ingestion_state.json so unchanged
    documents are skipped.
  * Vectors belonging to a changed or deleted PDF are deleted before upsert,
    so stale chunks can never be retrieved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader  # noqa: E402
from langchain_huggingface import HuggingFaceEmbeddings  # noqa: E402
from langchain_pinecone import PineconeVectorStore  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
STATE_FILE = BASE_DIR / ".ingestion_state.json"

INDEX_NAME = "medical-chatbot"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384  # all-MiniLM-L6-v2 output size

CHUNK_SIZE = 500
CHUNK_OVERLAP = 20

UPSERT_BATCH_SIZE = 100  # vectors per Pinecone request


# --------------------------------------------------------------------------
# Terminal output helpers
# --------------------------------------------------------------------------
def step(n: int, title: str) -> None:
    print(f"\n[{n}/6] {title}")
    print("-" * 60)


def ok(msg: str) -> None:
    print(f"  OK   {msg}")


def info(msg: str) -> None:
    print(f"       {msg}")


def warn(msg: str) -> None:
    print(f"  WARN {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL {msg}")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    """Content fingerprint of a file. Cheap (1.9 MB) and change-sensitive."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_key(relative_path: str) -> str:
    """Short, stable, filename-safe prefix used to build chunk IDs."""
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]


def chunk_id(relative_path: str, page: int, position: int) -> str:
    """Deterministic Pinecone ID.

    Same file + same page + same position  ->  same ID  ->  upsert replaces.
    Guarantees idempotency: running ingestion twice does not double the index.
    """
    return f"{source_key(relative_path)}-p{page:04d}-c{position:04d}"


def warn_if_env_index_mismatch() -> None:
    import os

    env_index = os.getenv("INDEX_NAME")
    if env_index and env_index != INDEX_NAME:
        warn(
            f".env sets INDEX_NAME={env_index!r} but this script targets "
            f"{INDEX_NAME!r}. Using {INDEX_NAME!r} (hardcoded on purpose)."
        )


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            warn(f"{STATE_FILE.name} is corrupt; treating all files as new.")
    return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def pinecone_stats(index_name: str) -> dict:
    from pinecone import Pinecone

    pc = Pinecone()
    return pc.Index(index_name).describe_index_stats()


# --------------------------------------------------------------------------
# Pipeline stages
# --------------------------------------------------------------------------
def discover_pdfs(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("*.pdf"))


def load_pdfs(pdf_paths: list[Path]) -> list:
    """One Document per PDF page. This is the 'pages' stage."""
    documents = []
    for pdf in pdf_paths:
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        documents.extend(pages)
        info(f"{pdf.name}: {len(pages)} pages")
    return documents


def split_documents(documents: list) -> list:
    """The 'chunks' stage. 500 chars with 20 chars of overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


def normalize_metadata(documents: list, data_dir: Path) -> list:
    """Rewrite metadata into portable, queryable, deterministic fields.

    PyPDFLoader records an absolute path in `source`, which differs per machine
    and across Windows/POSIX. We store a relative POSIX path instead so the
    frontend can cite sources reliably.

    We also drop producer/creator/moddate noise: Pinecone counts metadata
    toward the per-record size limit, and none of it is useful at query time.
    """
    keep = ("title", "author", "total_pages", "page_label")
    for doc in documents:
        original = Path(doc.metadata.get("source", "unknown"))
        try:
            relative = original.resolve().relative_to(data_dir.resolve())
            doc.metadata["source"] = relative.as_posix()
        except ValueError:
            doc.metadata["source"] = original.name

        extras = {k: doc.metadata[k] for k in keep if k in doc.metadata}
        page = doc.metadata.get("page", 0)
        doc.metadata = {
            "source": doc.metadata["source"],
            "source_key": source_key(doc.metadata["source"]),
            "page": int(page),
            "chunk_index": 0,  # filled in by build_chunk_records
            **extras,
        }
    return documents


def build_chunk_records(chunks: list) -> list:
    """Stamp each chunk with its stable position within its source file.

    This MUST run before IDs are generated. Without a unique position every
    chunk on a page collapses onto the same Pinecone ID and silently
    overwrites its siblings.
    """
    per_file_counter: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata["source"]
        position = per_file_counter.get(source, 0)
        per_file_counter[source] = position + 1
        chunk.metadata["chunk_index"] = position
    return chunks


def load_embeddings() -> HuggingFaceEmbeddings:
    """Load model WEIGHTS into RAM. This is not the same as embedding data."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        # NOTE: do NOT set show_progress_bar here. langchain_huggingface passes
        # its own show_progress_bar kwarg into SentenceTransformer.encode(),
        # so supplying it raises "got multiple values for keyword argument".
        encode_kwargs={"batch_size": 64},
    )


def source_aliases(rel_path: str) -> list[str]:
    """Every `source` value a previous run might have written.

    LangChain's default ingestion stored `data\\Medical_book.pdf` (Windows
    separators, relative to backend/). Those ids are random UUIDs, so they can
    neither be upserted over nor reliably matched by the normalized value.
    We therefore try each known alias explicitly when purging.
    """
    name = Path(rel_path).name
    return list(
        dict.fromkeys(
            [
                rel_path,
                name,
                f"data/{rel_path}",
                f"data\\{rel_path}",
                str((DATA_DIR / name)),
                str(Path(DATA_DIR / name).resolve()),
            ]
        )
    )


def delete_source(index_name: str, aliases: list[str]) -> int:
    """Remove every vector belonging to one source document. Returns rows purged."""
    from pinecone import Pinecone

    index = Pinecone().Index(index_name)
    purged = 0
    for alias in aliases:
        try:
            before = index.describe_index_stats()["total_vector_count"]
            index.delete(filter={"source": {"$eq": alias}})
            after = index.describe_index_stats()["total_vector_count"]
            purged += max(0, before - after)
            if before != after:
                info(f"purged {before - after} legacy vectors for source={alias!r}")
        except Exception as exc:  # noqa: BLE001
            # A 404 "Namespace not found" simply means the namespace is
            # already empty, which is exactly the state we want.
            if "Namespace not found" in str(exc):
                continue
            warn(f"purge failed for {alias!r}: {exc}")
    return purged


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Pinecone.")
    parser.add_argument("--force", action="store_true", help="Re-ingest all files.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe the whole index first. Required once to migrate off "
        "legacy random-UUID ids that cannot be matched or replaced.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report changes without writing."
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    started = time.time()

    print("=" * 60)
    print("Medical RAG - Document Ingestion")
    print("=" * 60)
    print(f"  data dir : {data_dir}")
    print(f"  index    : {INDEX_NAME}")
    print(f"  model    : {EMBEDDING_MODEL}")
    print(
        f"  chunking : size={CHUNK_SIZE} overlap={CHUNK_OVERLAP}"
    )
    warn_if_env_index_mismatch()

    # -- 1. Discover -------------------------------------------------------
    step(1, "Discovering PDFs")
    pdf_paths = discover_pdfs(data_dir)
    print(f"  {len(pdf_paths)} PDF(s) found")
    for pdf in pdf_paths:
        info(f"- {pdf.name} ({pdf.stat().st_size / 1024:.0f} KB)")
    if not pdf_paths:
        fail(f"No PDFs in {data_dir}. Nothing to ingest.")
        return 1

    # -- 2. Change detection ------------------------------------------------
    step(2, "Detecting changed documents")
    state = load_state()
    to_ingest: list[Path] = []
    for pdf in pdf_paths:
        rel = pdf.relative_to(data_dir).as_posix()
        digest = sha256_of(pdf)
        previous = state.get(rel, {}).get("sha256")
        if args.reset:
            info(f"{rel}: --reset, re-ingesting")
        elif previous == digest and not args.force:
            info(f"{rel}: unchanged, skipping")
            continue
        elif previous is None:
            info(f"{rel}: new document")
        elif args.force:
            info(f"{rel}: --force, re-ingesting")
        else:
            info(f"{rel}: content changed, re-ingesting")
        state[rel] = {"sha256": digest}
        to_ingest.append(pdf)

    # Vectors for PDFs that no longer exist on disk.
    orphaned = [rel for rel in state if rel not in {p.relative_to(data_dir).as_posix() for p in pdf_paths}]
    if orphaned:
        info(f"orphaned sources to purge: {', '.join(orphaned)}")

    if not to_ingest and not orphaned:
        ok("Nothing changed. Index already up to date.")
        print(f"\nDone in {time.time() - started:.1f}s. No writes performed.")
        return 0

    if args.dry_run:
        print(
            f"\nDRY RUN - would ingest {len(to_ingest)} file(s) and purge "
            f"{len(orphaned)} orphaned source(s). No writes performed."
        )
        return 0

    # -- 3. Load + chunk ----------------------------------------------------
    step(3, "Loading and chunking")
    documents = load_pdfs(pdf_paths)
    print(f"  {len(documents)} pages extracted")
    documents = normalize_metadata(documents, data_dir)
    chunks = split_documents(documents)
    chunks = build_chunk_records(chunks)
    print(f"  {len(chunks)} chunks produced")
    info(f"avg chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
    info(f"chunk ids are unique: {len({chunk_id(c.metadata['source'], c.metadata.get('page', 0), c.metadata['chunk_index']) for c in chunks})} / {len(chunks)}")

    # -- 4. Embeddings ------------------------------------------------------
    step(4, "Loading embedding model")
    embeddings = load_embeddings()
    print(f"  {EMBEDDING_MODEL} loaded successfully")
    info(f"dimension: {EXPECTED_DIMENSION}")

    # Guard against ingesting into an index built with a different model.
    try:
        stats = pinecone_stats(INDEX_NAME)
        dim = getattr(stats, "dimension", None) or stats.get("dimension")
        if dim and int(dim) != EXPECTED_DIMENSION:
            fail(
                f"Index '{INDEX_NAME}' has dimension {dim}, but "
                f"{EMBEDDING_MODEL} produces {EXPECTED_DIMENSION}."
            )
            info("Recreate the index with dimension 384, or switch models.")
            return 2
        info(f"index dimension check passed ({dim})")
    except Exception as exc:  # noqa: BLE001
        warn(f"Could not verify index dimension: {exc}")

    # -- 5. Upsert ----------------------------------------------------------
    step(5, "Embedding chunks and uploading to Pinecone")

    if args.reset:
        from pinecone import Pinecone

        try:
            Pinecone().Index(INDEX_NAME).delete(delete_all=True)
            ok("index wiped (--reset): legacy random-UUID vectors removed")
        except Exception as exc:  # noqa: BLE001
            if "Namespace not found" in str(exc):
                ok("index already empty, nothing to wipe")
            else:
                raise

    rel_names = {p.relative_to(data_dir).as_posix() for p in to_ingest}
    targets = [c for c in chunks if c.metadata["source"] in rel_names]
    target_ids = [
        chunk_id(c.metadata["source"], c.metadata.get("page", 0), c.metadata["chunk_index"])
        for c in targets
    ]

    # Purge stale vectors first. A changed PDF may have shrunk, and a removed
    # PDF must not leave orphans behind.
    for rel in orphaned:
        purged = delete_source(INDEX_NAME, source_aliases(rel))
        info(f"orphaned source purged ({purged} vectors): {rel}")
    for pdf in to_ingest:
        rel = pdf.relative_to(data_dir).as_posix()
        delete_source(INDEX_NAME, source_aliases(rel))

    if targets:
        print(f"  embedding {len(targets)} chunks (this is the slow step)...")
        PineconeVectorStore.from_documents(
            documents=targets,
            embedding=embeddings,
            index_name=INDEX_NAME,
            ids=target_ids,
            batch_size=UPSERT_BATCH_SIZE,
        )

    # -- 6. Verify ----------------------------------------------------------
    step(6, "Verification")
    try:
        stats = pinecone_stats(INDEX_NAME)
        total = getattr(stats, "total_vector_count", None) or stats.get("total_vector_count")
        print(f"  Pinecone upload completed: {total} vectors in '{INDEX_NAME}'")
    except Exception as exc:  # noqa: BLE001
        warn(f"Could not read index stats: {exc}")

    save_state(state)
    ok("Ingestion state saved")
    print(f"\nDone in {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)