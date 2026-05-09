"""
TableAgent
==========
Specialized agent for extracting and querying structured data,
tables, figures, and numerical information from PDFs.
"""

from __future__ import annotations
import json
import logging

logger = logging.getLogger(__name__)


class TableAgent:
    def __init__(self, vector_store, llm, settings: dict):
        self.vector_store = vector_store
        self.llm = llm
        self.settings = settings

    def run(self, query: str, history: list[dict]) -> dict:
        """Answer questions about tables and structured data."""
        docs = self.vector_store.similarity_search(
            f"table figure data statistics numbers {query}", k=8
        )
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""You are a data analysis expert. Extract and analyze structured data from the document.

DOCUMENT CONTENT:
{context}

USER QUERY: {query}

Instructions:
- Identify any tables, figures, statistics, or structured data
- Present numerical data clearly (use markdown tables if helpful)
- Calculate derived metrics if asked (averages, totals, trends)
- Highlight key data points

ANALYSIS:"""

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

    def extract_all(self) -> list[dict]:
        """Extract all tables from the knowledge base."""
        docs = self.vector_store.similarity_search(
            "table column row data percentage number statistics", k=15
        )
        tables = []
        for doc in docs:
            if any(kw in doc.page_content.lower() for kw in ["table", "|", "%", "figure"]):
                tables.append({
                    "content": doc.page_content,
                    "page": doc.metadata.get("page", "?"),
                    "source": doc.metadata.get("source", "unknown"),
                })
        return tables
