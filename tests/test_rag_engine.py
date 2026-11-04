"""
Offline tests for the retrieval layer — no LLM/network required.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_engine import FaissVectorStore, load_all_documents  # noqa: E402


def test_load_case_studies():
    docs = load_all_documents("data/casestudies")
    assert len(docs) >= 3
    sources = {d.metadata.get("source") for d in docs}
    assert {"powerco_churn", "nike_stock", "campaign_brands"} <= sources


def test_faiss_build_and_search(tmp_path):
    store = FaissVectorStore(persist_dir=str(tmp_path / "idx"))
    store.build_from_documents(load_all_documents("data/casestudies"))
    results = store.query("churn drivers and 20% discount", top_k=3)
    assert 1 <= len(results) <= 3
    assert all("metadata" in r and "distance" in r for r in results)
    # The churn pack should be the most relevant hit for a churn query.
    assert any((r["metadata"] or {}).get("source") == "powerco_churn" for r in results)


def test_search_requires_index():
    store = FaissVectorStore(persist_dir="does-not-exist-xyz")
    store.index = None
    with pytest.raises(ValueError):
        store.search(np.zeros((1, 384), dtype="float32"))
