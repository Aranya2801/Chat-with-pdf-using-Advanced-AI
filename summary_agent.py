"""
SummaryAgent
============
Generates structured summaries using:
  - Map-reduce summarization for long documents
  - Extractive + abstractive hybrid approach
  - Multiple output formats (bullets, executive, Q&A, mind map)
  - Section-aware summarization
"""

from __future__ import annotations
import logging
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


STYLE_PROMPTS = {
    "Executive Summary": """Create a professional executive summary with:
- **Overview** (2-3 sentences)
- **Key Findings** (3-5 bullet points)
- **Critical Insights** (numbered list)
- **Recommendations / Takeaways** (if applicable)
Use formal, concise language suitable for senior leadership.""",

    "Key Concepts": """Extract and explain the key concepts with:
- Concept name (bold)
- Definition (1-2 sentences)
- Relevance to the document
- Related concepts
Format as a well-organized reference guide.""",

    "Q&A Format": """Generate a comprehensive Q&A document:
- Anticipate the most important questions a reader would ask
- Provide detailed, accurate answers based on the content
- Include follow-up questions where appropriate
Format: **Q: [question]**\n**A: [answer]**""",

    "Bullet Points": """Create a structured bullet-point summary:
- Main topic bullets (•)
  - Supporting details (◦)
    - Specific examples or data (▪)
Cover all major sections and key information.""",

    "Mind Map": """Create a text-based mind map:
📌 CENTRAL TOPIC
├── 🔵 Main Branch 1
│   ├── Sub-topic 1.1
│   └── Sub-topic 1.2
├── 🟢 Main Branch 2
│   ├── Sub-topic 2.1
│   └── Sub-topic 2.2
└── 🔴 Main Branch 3
    └── Sub-topic 3.1
Use emojis and ASCII tree structure.""",
}


class SummaryAgent:
    """Extractive + abstractive document summarization."""

    def __init__(self, vector_store, llm, settings: dict):
        self.vector_store = vector_store
        self.llm = llm
        self.settings = settings

    def run(self, query: str, history: list[dict]) -> dict:
        """Summarize in response to a query."""
        docs = self.vector_store.similarity_search(query, k=12)
        context = "\n\n".join(d.page_content for d in docs)
        summary = self._summarize(query, context)
        return {
            "answer": summary,
            "sources": [],
            "citations": [],
        }

    def generate_notes(self, style: str = "Executive Summary") -> str:
        """Generate full-document notes in the requested style."""
        # Retrieve a broad sample of the document
        docs = self.vector_store.similarity_search(
            "main content overview introduction conclusion", k=20
        )
        # Map step: summarize each chunk
        chunk_summaries = []
        for i, doc in enumerate(docs):
            chunk_sum = self._map_chunk(doc.page_content, i)
            chunk_summaries.append(chunk_sum)

        # Reduce step: combine all summaries
        combined = "\n\n".join(chunk_summaries)
        style_instruction = STYLE_PROMPTS.get(style, STYLE_PROMPTS["Executive Summary"])

        reduce_prompt = f"""{style_instruction}

Based on these document sections:

{combined}

Generate the requested notes now:"""

        response = self.llm.invoke(reduce_prompt)
        return response.content

    def _map_chunk(self, chunk: str, index: int) -> str:
        prompt = f"""Summarize the key information from this document section in 2-3 sentences:

{chunk}

Summary:"""
        try:
            response = self.llm.invoke(prompt)
            return f"[Section {index+1}] {response.content.strip()}"
        except Exception:
            return chunk[:300]

    def _summarize(self, query: str, context: str) -> str:
        prompt = f"""You are a precise summarization AI.

USER REQUEST: {query}

DOCUMENT CONTENT:
{context}

Provide a comprehensive, well-structured summary that directly addresses the user's request.
Use markdown formatting for clarity."""

        response = self.llm.invoke(prompt)
        return response.content
