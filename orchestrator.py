"""
Multi-Agent Orchestrator
========================
Routes queries through specialized sub-agents:
  • RetrievalAgent  — hybrid BM25 + dense vector search + reranking
  • ReasoningAgent  — chain-of-thought multi-hop QA
  • SummaryAgent    — extractive + abstractive summarization
  • CitationAgent   — grounded answer with page/section citations
  • TableAgent      — structured data extraction from tables/figures
"""

from __future__ import annotations
import time
import json
import logging
from typing import Any
from openai import OpenAI
from langchain_openai import ChatOpenAI

from src.agents.retrieval_agent import RetrievalAgent
from src.agents.reasoning_agent import ReasoningAgent
from src.agents.summary_agent import SummaryAgent
from src.agents.citation_agent import CitationAgent
from src.agents.table_agent import TableAgent
from src.utils.query_classifier import classify_query
from src.utils.memory import ConversationMemory

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Central orchestrator that routes queries to the best agent.
    Uses intent classification + confidence routing.
    """

    AGENT_MAP = {
        "factual":      "retrieval",
        "reasoning":    "reasoning",
        "summary":      "summary",
        "table":        "table",
        "comparison":   "reasoning",
        "definition":   "retrieval",
        "procedure":    "reasoning",
    }

    def __init__(self, vector_store, settings: dict):
        self.vector_store = vector_store
        self.settings = settings
        self.memory = ConversationMemory(window=settings.get("memory_window", 10))
        self.llm = ChatOpenAI(
            model=settings.get("model", "gpt-4o"),
            temperature=settings.get("temperature", 0.1),
            streaming=True,
        )
        self._init_agents()
        logger.info("AgentOrchestrator initialized with model=%s", settings.get("model"))

    def _init_agents(self):
        cfg = dict(vector_store=self.vector_store, llm=self.llm, settings=self.settings)
        self.agents = {
            "retrieval": RetrievalAgent(**cfg),
            "reasoning": ReasoningAgent(**cfg),
            "summary":   SummaryAgent(**cfg),
            "citation":  CitationAgent(**cfg),
            "table":     TableAgent(**cfg),
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, query: str, stream: bool = True):
        """
        Main entry point for user queries.
        Returns generator (streaming) or dict (non-streaming).
        """
        t0 = time.perf_counter()
        intent, confidence = classify_query(query)
        agent_name = self.AGENT_MAP.get(intent, "retrieval")

        logger.info("Query intent=%s conf=%.2f → agent=%s", intent, confidence, agent_name)

        # Build context from memory
        history = self.memory.get_history()

        # Primary agent response
        primary_agent = self.agents[agent_name]
        result = primary_agent.run(query=query, history=history)

        # Always enrich with citations if enabled
        if self.settings.get("citation_mode") and agent_name != "citation":
            result = self.agents["citation"].enrich(result)

        # Update memory
        self.memory.add(role="user", content=query)
        self.memory.add(role="assistant", content=result["answer"])

        elapsed = time.perf_counter() - t0
        result["meta"] = {
            "agent": agent_name,
            "intent": intent,
            "confidence": confidence,
            "latency_ms": round(elapsed * 1000),
            "model": self.settings.get("model"),
            "sources_count": len(result.get("sources", [])),
        }

        if stream:
            return self._stream_result(result)
        return result

    def generate_notes(self, style: str = "Executive Summary") -> str:
        """Generate structured notes from the entire knowledge base."""
        return self.agents["summary"].generate_notes(style=style)

    def generate_quiz(self, num_questions: int = 5) -> list[dict]:
        """Auto-generate quiz questions from the documents."""
        return self.agents["reasoning"].generate_quiz(num_questions=num_questions)

    def extract_tables(self) -> list[dict]:
        """Extract all tables and structured data from documents."""
        return self.agents["table"].extract_all()

    # ── Streaming ──────────────────────────────────────────────────────────────

    def _stream_result(self, result: dict):
        """Yield answer tokens then metadata."""
        answer = result.get("answer", "")
        # Simulate token streaming for display
        words = answer.split()
        buffer = ""
        for i, word in enumerate(words):
            buffer += word + " "
            if i % 3 == 0:
                yield {"type": "token", "content": buffer}
                buffer = ""
                time.sleep(0.005)
        if buffer:
            yield {"type": "token", "content": buffer}

        yield {"type": "meta", "content": result.get("meta", {})}
        yield {"type": "sources", "content": result.get("sources", [])}
        yield {"type": "citations", "content": result.get("citations", [])}
