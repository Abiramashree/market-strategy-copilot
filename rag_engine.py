"""
rag_engine.py
Retrieval over the case-study knowledge packs (Markdown) for the Market Strategy Copilot.

Pipeline: Markdown case studies -> chunk -> embed (SentenceTransformers, local) ->
FAISS index -> semantic search -> open-source/LLM summarization.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - older LangChain
    from langchain.text_splitter import RecursiveCharacterTextSplitter

import config
from llm import make_llm


def load_all_documents(data_dir: str = config.DATA_DIR) -> List[Document]:
    """Load the Markdown case-study packs as LangChain Documents (one per file)."""
    data_path = Path(data_dir).resolve()
    documents: List[Document] = []
    for md in sorted(data_path.glob("**/*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                documents.append(Document(page_content=text, metadata={"source": md.stem}))
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to read {md}: {exc}")
    return documents


class EmbeddingPipeline:
    """Chunk documents and embed them with a local SentenceTransformer."""

    def __init__(
        self,
        model_name: str = config.EMBED_MODEL,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        )
        return splitter.split_documents(documents)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, show_progress_bar=False), dtype="float32")


class FaissVectorStore:
    """Persistent FAISS store with per-vector metadata (source + text)."""

    def __init__(
        self,
        persist_dir: str = config.PERSIST_DIR,
        embedding_model: str = config.EMBED_MODEL,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ):
        self.persist_dir = persist_dir
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(embedding_model)

    @property
    def _index_path(self) -> str:
        return os.path.join(self.persist_dir, "faiss.index")

    @property
    def _meta_path(self) -> str:
        return os.path.join(self.persist_dir, "metadata.pkl")

    def build_from_documents(self, documents: List[Document]) -> None:
        pipeline = EmbeddingPipeline(self.embedding_model, self.chunk_size, self.chunk_overlap)
        chunks = pipeline.chunk_documents(documents)
        embeddings = pipeline.embed_texts([c.page_content for c in chunks])
        metadatas = [
            {"text": c.page_content, "source": c.metadata.get("source", "")} for c in chunks
        ]
        self.add_embeddings(embeddings, metadatas)
        self.save()

    def add_embeddings(self, embeddings: np.ndarray, metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array of shape (n_vectors, dim).")
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)

    def save(self) -> None:
        if self.index is None:
            raise ValueError("No FAISS index to save.")
        faiss.write_index(self.index, self._index_path)
        with open(self._meta_path, "wb") as fh:
            pickle.dump(self.metadata, fh)

    def load(self) -> None:
        if not (os.path.exists(self._index_path) and os.path.exists(self._meta_path)):
            raise FileNotFoundError(f"Missing index/metadata in {self.persist_dir}. Build first.")
        self.index = faiss.read_index(self._index_path)
        with open(self._meta_path, "rb") as fh:
            self.metadata = pickle.load(fh)

    def search(self, query_embedding: np.ndarray, top_k: int = config.TOP_K) -> List[Dict[str, Any]]:
        if self.index is None:
            raise ValueError("FAISS index is not loaded or built.")
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding.astype("float32"), top_k)
        results: List[Dict[str, Any]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx] if 0 <= idx < len(self.metadata) else None
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = config.TOP_K) -> List[Dict[str, Any]]:
        query_emb = self.model.encode([query_text]).astype("float32")
        return self.search(query_emb, top_k=top_k)

    def ensure_built(self, data_dir: str = config.DATA_DIR) -> None:
        if os.path.exists(self._index_path) and os.path.exists(self._meta_path):
            self.load()
        else:
            self.build_from_documents(load_all_documents(data_dir))


class RAGSearch:
    """Retrieve relevant case-study context and summarize it with the configured LLM."""

    def __init__(
        self,
        persist_dir: str = config.PERSIST_DIR,
        embedding_model: str = config.EMBED_MODEL,
        llm_model: str = config.SUMMARY_MODEL,
    ):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.vectorstore.ensure_built()
        self.llm = make_llm(model=llm_model, temperature=0)

    def search_and_summarize(self, query: str, top_k: int = config.TOP_K) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        chunks = [r["metadata"]["text"] for r in results if r.get("metadata")]
        context = "\n\n".join(chunks) if chunks else "[No relevant case-study context found.]"
        prompt = (
            "You are a market-strategy analyst. Using ONLY the case-study excerpts below, "
            "answer the question concisely with concrete evidence (numbers, drivers, caveats). "
            f"\n\nExcerpts:\n{context}\n\nQuestion: {query}"
        )
        return self.llm.invoke(prompt).content
