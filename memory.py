"""
ConversationMemory
==================
Sliding-window conversation memory with optional summarization
for very long conversations (>N turns).
"""

from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path


class ConversationMemory:
    """Fixed-window conversation history with persistence."""

    def __init__(self, window: int = 10):
        self.window = window
        self._history: list[dict] = []

    def add(self, role: str, content: str):
        self._history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Keep only the last N*2 turns (N user + N assistant)
        if len(self._history) > self.window * 2:
            self._history = self._history[-(self.window * 2):]

    def get_history(self) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in self._history]

    def get_full_history(self) -> list[dict]:
        return list(self._history)

    def clear(self):
        self._history.clear()

    def save(self, path: str):
        Path(path).write_text(json.dumps(self._history, indent=2))

    def load(self, path: str):
        if Path(path).exists():
            self._history = json.loads(Path(path).read_text())

    def __len__(self):
        return len(self._history)
