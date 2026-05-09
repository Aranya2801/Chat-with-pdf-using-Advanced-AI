"""
QueryClassifier
===============
Classifies user queries into intent categories for agent routing.
Uses a lightweight rule-based + LLM hybrid approach.

Intent categories:
  factual     → RetrievalAgent
  reasoning   → ReasoningAgent
  summary     → SummaryAgent
  table       → TableAgent
  comparison  → ReasoningAgent
  definition  → RetrievalAgent
  procedure   → ReasoningAgent
"""

from __future__ import annotations
import re

# ── Rule-based patterns (fast, no LLM call) ───────────────────────────────────

INTENT_PATTERNS = {
    "summary": [
        r"\bsummar(y|ize|ise)\b",
        r"\boverview\b",
        r"\bbrief(ly)?\b",
        r"\bgist\b",
        r"\btldr\b",
        r"\bmain (points?|ideas?|topics?)\b",
        r"\bwhat (is|are) (the )?(document|paper|report|text) about\b",
    ],
    "table": [
        r"\btable\b",
        r"\bchart\b",
        r"\bfigure\b",
        r"\bstatistic\b",
        r"\bnumber(s)?\b",
        r"\bdata\b",
        r"\bpercentage\b",
        r"\bcompare (the )?number\b",
        r"\bshow (me )?(the )?data\b",
    ],
    "comparison": [
        r"\bcompare\b",
        r"\bvs\.?\b",
        r"\bversus\b",
        r"\bdifference between\b",
        r"\bsimilar(ities)?\b",
        r"\bcontrast\b",
    ],
    "definition": [
        r"\bwhat (is|are|does)\b",
        r"\bdefine\b",
        r"\bmeaning of\b",
        r"\bexplain\b",
        r"\bdescribe\b",
    ],
    "procedure": [
        r"\bhow (to|do|can|should)\b",
        r"\bsteps?\b",
        r"\bprocess\b",
        r"\bmethod\b",
        r"\bprocedure\b",
        r"\binstruction\b",
    ],
    "reasoning": [
        r"\bwhy\b",
        r"\breason(s)?\b",
        r"\bcause\b",
        r"\bimplication\b",
        r"\bimpact\b",
        r"\bconsequence\b",
        r"\banalyz(e|sis)\b",
        r"\binfer\b",
        r"\bdeduc\b",
    ],
}


def classify_query(query: str) -> tuple[str, float]:
    """
    Classify query intent using rule-based matching.
    Returns (intent, confidence) where confidence ∈ [0, 1].
    """
    query_lower = query.lower().strip()
    scores: dict[str, float] = {}

    for intent, patterns in INTENT_PATTERNS.items():
        matches = sum(
            1 for p in patterns if re.search(p, query_lower)
        )
        if matches > 0:
            scores[intent] = matches / len(patterns)

    if not scores:
        return "factual", 0.5

    best_intent = max(scores, key=lambda k: scores[k])
    confidence = min(scores[best_intent] * 3, 1.0)  # scale up

    # Boost confidence for clear signals
    if scores.get(best_intent, 0) > 0.3:
        confidence = max(confidence, 0.8)

    return best_intent, round(confidence, 2)


def get_query_complexity(query: str) -> str:
    """
    Estimate query complexity: simple | medium | complex
    Based on length, conjunction count, and multi-part structure.
    """
    words = query.split()
    conjunctions = sum(1 for w in words if w.lower() in {"and", "also", "additionally", "furthermore", "moreover"})
    question_marks = query.count("?")

    if len(words) < 10 and conjunctions == 0:
        return "simple"
    if len(words) > 30 or conjunctions > 2 or question_marks > 2:
        return "complex"
    return "medium"
