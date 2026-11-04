"""
scripts/build_index.py
Pre-build the FAISS index from the case-study packs in DATA_DIR.

Run once before serving so the first query is fast:  python scripts/build_index.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from rag_engine import FaissVectorStore, load_all_documents  # noqa: E402


def main() -> None:
    docs = load_all_documents(config.DATA_DIR)
    if not docs:
        raise SystemExit(f"No case-study markdown found in {config.DATA_DIR!r}.")
    FaissVectorStore().build_from_documents(docs)
    print(f"Built FAISS index from {len(docs)} case-study docs -> {config.PERSIST_DIR}/")


if __name__ == "__main__":
    main()
