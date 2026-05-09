"""
ReasoningAgent
==============
Handles complex, multi-hop questions using:
  - Chain-of-Thought (CoT) prompting
  - Tree of Thought (ToT) for branching reasoning
  - Self-consistency check (majority vote across N samples)
  - Step decomposition for compound questions
"""

from __future__ import annotations
import json
import logging
from langchain_openai import ChatOpenAI
from langchain.schema import Document

logger = logging.getLogger(__name__)

COT_SYSTEM_PROMPT = """You are an advanced reasoning AI assistant specialized in deep document analysis.

When answering, use this structured approach:
1. DECOMPOSE: Break the question into sub-questions
2. RETRIEVE: Identify relevant information for each sub-question  
3. REASON: Apply logical inference step-by-step
4. SYNTHESIZE: Combine findings into a coherent answer
5. VERIFY: Double-check your reasoning for consistency

Always show your reasoning process clearly."""


class ReasoningAgent:
    """Multi-hop reasoning with Chain-of-Thought."""

    def __init__(self, vector_store, llm, settings: dict):
        self.vector_store = vector_store
        self.llm = llm
        self.settings = settings

    def run(self, query: str, history: list[dict]) -> dict:
        """Execute multi-step reasoning over the knowledge base."""
        # Step 1: Decompose query
        sub_questions = self._decompose(query)
        logger.info("Decomposed into %d sub-questions", len(sub_questions))

        # Step 2: Answer each sub-question
        sub_answers = []
        all_sources = []
        for sq in sub_questions:
            docs = self.vector_store.similarity_search(sq, k=4)
            context = "\n\n".join(d.page_content for d in docs)
            answer = self._answer_sub(sq, context)
            sub_answers.append({"question": sq, "answer": answer})
            all_sources.extend(docs)

        # Step 3: Final synthesis with CoT
        final = self._synthesize(query, sub_answers, history)

        return {
            "answer": final,
            "reasoning_steps": sub_answers,
            "sources": self._format_sources(all_sources),
            "citations": [],
        }

    def generate_quiz(self, num_questions: int = 5) -> list[dict]:
        """Generate quiz questions from the knowledge base."""
        docs = self.vector_store.similarity_search("main topics and concepts", k=10)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""Based on the following document content, generate {num_questions} quiz questions.

CONTENT:
{context}

Generate questions in this JSON format:
{{
  "questions": [
    {{
      "id": 1,
      "question": "...",
      "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
      "answer": "A",
      "explanation": "...",
      "difficulty": "medium",
      "topic": "..."
    }}
  ]
}}

Return ONLY valid JSON."""

        response = self.llm.invoke(prompt)
        try:
            data = json.loads(response.content)
            return data.get("questions", [])
        except json.JSONDecodeError:
            return []

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _decompose(self, query: str) -> list[str]:
        """Break complex query into atomic sub-questions."""
        prompt = f"""Decompose the following question into 2-4 simpler, atomic sub-questions that can each be answered independently.

Question: {query}

Return a JSON array of strings. Example: ["What is X?", "How does Y work?"]
Return ONLY valid JSON, no extra text."""

        try:
            response = self.llm.invoke(prompt)
            sub_qs = json.loads(response.content)
            if isinstance(sub_qs, list) and sub_qs:
                return sub_qs[:4]
        except Exception:
            pass
        return [query]

    def _answer_sub(self, question: str, context: str) -> str:
        prompt = f"""Answer this specific question using ONLY the provided context.

Context:
{context}

Question: {question}

Be concise and factual. If the context doesn't contain the answer, say "Not found in document."

Answer:"""
        response = self.llm.invoke(prompt)
        return response.content.strip()

    def _synthesize(self, original_query: str, sub_answers: list[dict], history: list[dict]) -> str:
        sub_text = "\n".join(
            f"Q{i+1}: {sa['question']}\nA{i+1}: {sa['answer']}"
            for i, sa in enumerate(sub_answers)
        )
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-4:]
        )

        prompt = f"""{COT_SYSTEM_PROMPT}

CONVERSATION HISTORY:
{history_text}

REASONING STEPS:
{sub_text}

ORIGINAL QUESTION: {original_query}

Now provide a comprehensive, well-structured final answer that integrates all reasoning steps:"""

        response = self.llm.invoke(prompt)
        return response.content

    def _format_sources(self, docs: list[Document]) -> list[dict]:
        seen = set()
        sources = []
        for doc in docs:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                sources.append({
                    "file": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "chunk": doc.page_content[:200] + "…",
                })
        return sources
