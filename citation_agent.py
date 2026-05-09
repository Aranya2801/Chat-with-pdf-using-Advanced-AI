"""
CitationAgent
=============
Enriches answers with precise page-level citations.
"""

from __future__ import annotations
import re
import logging

logger = logging.getLogger(__name__)


class CitationAgent:
    def __init__(self, vector_store, llm, settings: dict):
        self.vector_store = vector_store
        self.llm = llm
        self.settings = settings

    def enrich(self, result: dict) -> dict:
        """Add inline citations to an existing answer."""
        if not result.get("sources"):
            return result

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # Build citation map
        citation_block = "\n\n---\n**📚 Sources:**\n"
        for i, src in enumerate(sources[:5], 1):
            citation_block += (
                f"\n[{i}] **{src.get('file', 'Document')}**, "
                f"Page {src.get('page', '?')}"
            )
            if src.get("section"):
                citation_block += f" — *{src['section']}*"

        result["answer"] = answer + citation_block
        return result

    def run(self, query: str, history: list[dict]) -> dict:
        docs = self.vector_store.similarity_search(query, k=6)
        context = "\n\n".join(
            f"[Source {i+1}, Page {d.metadata.get('page','?')}]: {d.page_content}"
            for i, d in enumerate(docs)
        )
        prompt = f"""Answer the question with precise inline citations using [1], [2], etc.

SOURCES:
{context}

QUESTION: {query}

Provide a detailed answer with citations after each claim. Example: "The revenue grew by 25% [1]."

ANSWER:"""
        response = self.llm.invoke(prompt)
        sources = [
            {
                "file": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", "?"),
                "chunk": d.page_content[:200],
            }
            for d in docs
        ]
        return {"answer": response.content, "sources": sources, "citations": []}
