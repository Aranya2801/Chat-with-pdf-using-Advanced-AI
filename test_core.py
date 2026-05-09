"""
Test Suite — Chat-with-PDF Advanced AI
=======================================
Covers:
  - QueryClassifier intent detection
  - ConversationMemory sliding window
  - PDFProcessor text extraction
  - VectorStoreManager CRUD
  - Agent routing logic
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document

from src.utils.query_classifier import classify_query, get_query_complexity
from src.utils.memory import ConversationMemory


# ═══════════════════════════════════════════════════════════════
# QueryClassifier Tests
# ═══════════════════════════════════════════════════════════════

class TestQueryClassifier:
    def test_summary_intent(self):
        intent, conf = classify_query("Can you summarize this document?")
        assert intent == "summary"
        assert conf > 0.5

    def test_table_intent(self):
        intent, conf = classify_query("Show me all the tables and statistics")
        assert intent == "table"

    def test_reasoning_intent(self):
        intent, conf = classify_query("Why did the experiment fail?")
        assert intent == "reasoning"

    def test_comparison_intent(self):
        intent, conf = classify_query("Compare the two approaches described")
        assert intent == "comparison"

    def test_definition_intent(self):
        intent, conf = classify_query("What is transformer architecture?")
        assert intent == "definition"

    def test_procedure_intent(self):
        intent, conf = classify_query("How do I implement this step by step?")
        assert intent == "procedure"

    def test_fallback_intent(self):
        intent, conf = classify_query("xyzzy random gibberish query")
        assert intent == "factual"
        assert 0 < conf <= 1.0

    def test_confidence_range(self):
        for query in ["summarize", "what is", "why", "compare", "show table"]:
            _, conf = classify_query(query)
            assert 0.0 <= conf <= 1.0, f"Confidence out of range for: {query}"

    def test_complexity_simple(self):
        assert get_query_complexity("What is X?") == "simple"

    def test_complexity_complex(self):
        q = "What are the causes and implications of X, and how does it compare to Y and Z?"
        assert get_query_complexity(q) == "complex"


# ═══════════════════════════════════════════════════════════════
# ConversationMemory Tests
# ═══════════════════════════════════════════════════════════════

class TestConversationMemory:
    def test_add_and_retrieve(self):
        mem = ConversationMemory(window=5)
        mem.add("user", "Hello")
        mem.add("assistant", "Hi there!")
        history = mem.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_sliding_window(self):
        mem = ConversationMemory(window=3)
        for i in range(10):
            mem.add("user", f"Message {i}")
            mem.add("assistant", f"Reply {i}")
        # Window of 3 = 6 messages max
        assert len(mem.get_history()) <= 6

    def test_clear(self):
        mem = ConversationMemory()
        mem.add("user", "test")
        mem.clear()
        assert len(mem) == 0

    def test_len(self):
        mem = ConversationMemory()
        assert len(mem) == 0
        mem.add("user", "hi")
        assert len(mem) == 1

    def test_history_format(self):
        mem = ConversationMemory()
        mem.add("user", "question")
        h = mem.get_history()
        assert "role" in h[0]
        assert "content" in h[0]
        assert "timestamp" not in h[0]  # stripped in get_history

    def test_full_history_has_timestamp(self):
        mem = ConversationMemory()
        mem.add("user", "question")
        h = mem.get_full_history()
        assert "timestamp" in h[0]


# ═══════════════════════════════════════════════════════════════
# PDFProcessor Tests
# ═══════════════════════════════════════════════════════════════

class TestPDFProcessor:
    def test_init(self):
        from src.utils.pdf_processor import PDFProcessor
        processor = PDFProcessor({"chunk_size": 500, "chunk_overlap": 50})
        assert processor.chunk_size == 500
        assert processor.chunk_overlap == 50

    def test_hash_deterministic(self):
        from src.utils.pdf_processor import PDFProcessor
        processor = PDFProcessor({})
        h1 = processor._hash("hello world")
        h2 = processor._hash("hello world")
        assert h1 == h2

    def test_hash_unique(self):
        from src.utils.pdf_processor import PDFProcessor
        processor = PDFProcessor({})
        h1 = processor._hash("text one")
        h2 = processor._hash("text two")
        assert h1 != h2

    def test_metadata_summary_empty(self):
        from src.utils.pdf_processor import PDFProcessor
        processor = PDFProcessor({})
        assert processor.get_metadata_summary([]) == {}

    def test_metadata_summary(self):
        from src.utils.pdf_processor import PDFProcessor
        processor = PDFProcessor({})
        docs = [
            Document(page_content="Hello world", metadata={"source": "test.pdf", "page": 1}),
            Document(page_content="More text here", metadata={"source": "test.pdf", "page": 2}),
        ]
        summary = processor.get_metadata_summary(docs)
        assert summary["total_chunks"] == 2
        assert "test.pdf" in summary["files"]


# ═══════════════════════════════════════════════════════════════
# VectorStoreManager Tests
# ═══════════════════════════════════════════════════════════════

class TestVectorStoreManager:
    def test_empty_search_returns_empty(self):
        from src.vectorstore.vector_store_manager import VectorStoreManager
        with patch("src.vectorstore.vector_store_manager.OpenAIEmbeddings"):
            manager = VectorStoreManager({"vector_backend": "faiss"})
            results = manager.similarity_search("anything")
            assert results == []

    def test_get_stats_empty(self):
        from src.vectorstore.vector_store_manager import VectorStoreManager
        with patch("src.vectorstore.vector_store_manager.OpenAIEmbeddings"):
            manager = VectorStoreManager({"vector_backend": "faiss"})
            stats = manager.get_stats()
            assert stats["indexed"] == 0
            assert stats["backend"] == "faiss"


# ═══════════════════════════════════════════════════════════════
# Agent Routing Tests
# ═══════════════════════════════════════════════════════════════

class TestAgentRouting:
    def test_intent_to_agent_mapping(self):
        from src.agents.orchestrator import AgentOrchestrator
        mapping = AgentOrchestrator.AGENT_MAP
        assert mapping["summary"] == "summary"
        assert mapping["reasoning"] == "reasoning"
        assert mapping["table"] == "table"
        assert mapping["factual"] == "retrieval"

    def test_all_intents_have_agent(self):
        from src.agents.orchestrator import AgentOrchestrator
        for intent in ["factual", "reasoning", "summary", "table", "comparison"]:
            assert intent in AgentOrchestrator.AGENT_MAP


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
