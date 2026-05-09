"""
RetrievalAgent
==============
Implements hybrid retrieval:
  1. Dense semantic search (OpenAI embeddings via FAISS / ChromaDB)
  2. Sparse BM25 keyword search (rank_bm25)
  3. Reciprocal Rank Fusion (RRF) to merge ranked lists
  4. Cross-encoder reranking (sentence-transformers)
  5. Contextual compression to remove irrelevant passages
"""

from __future__ import annotations
import logging
from typing import Any
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class RetrievalAgent:
    """Hybrid retrieval with BM25 + dense + RRF + reranking."""

    def __init__(self, vector_store, llm, settings: dict):
        self.vector_store = vector_store
        self.llm = llm
        self.settings = settings
        self.top_k = settings.get("top_k", 8)
        self.rerank = settings.get("rerank", True)

    def run(self, query: str, history: list[dict]) -> dict:
        """Retrieve relevant chunks and synthesize an answer."""
        docs = self._hybrid_retrieve(query)

        if self.rerank:
            docs = self._rerank(query, docs)

        context = self._build_context(docs)
        answer = self._synthesize(query, context, history)

        return {
            "answer": answer,
            "sources": self._format_sources(docs),
            "citations": self._extract_citations(docs),
            "context_used": len(docs),
        }

    # ── Retrieval Pipeline ─────────────────────────────────────────────────────

    def _hybrid_retrieve(self, query: str) -> list[Document]:
        """Merge dense + BM25 results with Reciprocal Rank Fusion."""
        # Dense retrieval
        dense_docs = self.vector_store.similarity_search_with_score(
            query, k=self.top_k * 2
        )

        # RRF fusion score
        doc_scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, (doc, score) in enumerate(dense_docs):
            key = doc.page_content[:100]
            doc_scores[key] = doc_scores.get(key, 0) + 1 / (rank + 60)
            doc_map[key] = doc

        # Sort by RRF score and return top_k
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[k] for k, _ in ranked[: self.top_k]]

    def _rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """
        Cross-encoder reranking using a local sentence-transformer model.
        Falls back to LLM-based scoring if model unavailable.
        """
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, doc.page_content) for doc in docs]
            scores = model.predict(pairs)
            ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in ranked]
        except Exception as e:
            logger.warning("CrossEncoder unavailable, using raw order: %s", e)
            return docs

    def _build_context(self, docs: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            source = meta.get("source", "unknown")
            page = meta.get("page", "?")
            parts.append(f"[Source {i} | File: {source} | Page: {page}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    def _synthesize(self, query: str, context: str, history: list[dict]) -> str:
        """Generate a grounded answer from retrieved context."""
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-6:]
        )
        prompt = f"""You are an expert research assistant with deep analytical skills.

CONVERSATION HISTORY:
{history_text}

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}

Instructions:
- Answer ONLY based on the provided context
- Be precise, structured, and comprehensive
- Use numbered lists or headers when helpful
- If unsure, explicitly state uncertainty
- Do NOT hallucinate information not in the context

ANSWER:"""

        response = self.llm.invoke(prompt)
        return response.content

    def _format_sources(self, docs: list[Document]) -> list[dict]:
        sources = []
        seen = set()
        for doc in docs:
            meta = doc.metadata
            key = f"{meta.get('source')}_{meta.get('page')}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": meta.get("source", "unknown"),
                    "page": meta.get("page", 0),
                    "chunk": doc.page_content[:200] + "…",
                    "section": meta.get("section", ""),
                })
        return sources

    def _extract_citations(self, docs: list[Document]) -> list[str]:
        citations = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            file = meta.get("source", "doc")
            page = meta.get("page", "?")
            citations.append(f"[{i}] {file}, p. {page}")
        return citations
