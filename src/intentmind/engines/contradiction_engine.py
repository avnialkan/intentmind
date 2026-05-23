"""
Contradiction Engine — Detects and resolves conflicting memories.

When a new chunk is ingested, this engine scans existing chunks that share
the same intent nodes. If two chunks are topically related but contain
contradictory information, the engine flags the conflict and applies a
resolution strategy.

Resolution strategies:
  - recency:       Newer chunk wins (default). Older chunk gets noise penalty.
  - reinforcement: Higher reinforcement_count wins.
  - coexist:       Both survive but contradiction is logged for transparency.

Detection is hybrid:
  - Deterministic heuristics run always (fast, zero cost).
  - Optional LLM verification can be enabled for higher accuracy.
"""
from __future__ import annotations

import re
import time
from typing import List, Optional

from ..store import MemoryStore
from ..models import Chunk
from ..embeddings import BaseEmbedder, cosine_similarity


# ------------------------------------------------------------------ #
#  Heuristic contradiction signals                                    #
# ------------------------------------------------------------------ #

# Negation markers (multilingual)
_NEGATION_PATTERNS = [
    # English
    r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bneither\b", r"\bnor\b",
    r"\bwon'?t\b", r"\bcan'?t\b", r"\bdon'?t\b", r"\bdoesn'?t\b",
    r"\bdidn'?t\b", r"\bisn'?t\b", r"\baren'?t\b", r"\bwasn'?t\b",
    r"\bweren'?t\b", r"\bshouldn'?t\b", r"\bcouldn'?t\b", r"\bwouldn'?t\b",
    r"\bfalse\b", r"\bwrong\b", r"\bincorrect\b", r"\bcancelled\b",
    r"\bcanceled\b", r"\bpostponed\b",
    # Turkish
    r"\bdeğil\b", r"\byok\b", r"\basla\b", r"\bhiç\b",
    r"\biptal\b", r"\bertelendi\b", r"\byanlış\b",
]
_NEGATION_RE = re.compile("|".join(_NEGATION_PATTERNS), re.IGNORECASE | re.UNICODE)

# Temporal conflict patterns (dates, days, times)
_TEMPORAL_RE = re.compile(
    r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|pazartesi|salı|çarşamba|perşembe|cuma|cumartesi|pazar"
    r"|january|february|march|april|may|june|july|august|september|october|november|december"
    r"|ocak|şubat|mart|nisan|mayıs|haziran|temmuz|ağustos|eylül|ekim|kasım|aralık"
    r"|\d{1,2}[:/]\d{2}"  # times like 14:00 or 2:30
    r"|\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})\b",  # dates
    re.IGNORECASE | re.UNICODE,
)

# Numeric value pattern
_NUMERIC_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:%|TL|USD|EUR|₺|\$|€|kg|km|m|cm)?\b")


class ContradictionEngine:
    """Detects and resolves contradictions between memory chunks."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: BaseEmbedder,
        strategy: str = "recency",
        similarity_floor: float = 0.55,
        similarity_ceiling: float = 0.88,
        noise_penalty: float = 0.30,
        llm_client=None,
        llm_model: str | None = None,
    ):
        self.store = store
        self.embedder = embedder
        self.strategy = strategy
        self.similarity_floor = similarity_floor
        self.similarity_ceiling = similarity_ceiling
        self.noise_penalty = noise_penalty
        self.llm_client = llm_client
        self.llm_model = llm_model
        self._contradiction_log: List[dict] = []

    # ------------------------------------------------------------------ #
    #  PUBLIC API                                                         #
    # ------------------------------------------------------------------ #

    def check_and_resolve(self, new_chunk: Chunk) -> List[dict]:
        """
        Check a newly ingested chunk against existing memories that share
        the same intent nodes. Returns a list of detected contradictions
        with resolution metadata.
        """
        contradictions: List[dict] = []

        for intent_id in new_chunk.intent_ids:
            existing_chunks = self.store.get_chunks_by_intent(intent_id)
            for old_chunk in existing_chunks:
                if old_chunk.chunk_id == new_chunk.chunk_id:
                    continue

                sim = cosine_similarity(new_chunk.embedding, old_chunk.embedding)

                # Too similar = duplicate (handled by consolidation), too different = unrelated
                if not (self.similarity_floor < sim < self.similarity_ceiling):
                    continue

                contradiction_type = self._detect_contradiction(
                    new_chunk.text, old_chunk.text
                )
                if not contradiction_type:
                    continue

                # Optional LLM verification for borderline cases
                if self.llm_client and contradiction_type == "possible_negation":
                    if not self._llm_verify(new_chunk.text, old_chunk.text):
                        continue

                resolution = self._resolve(new_chunk, old_chunk, contradiction_type)
                record = {
                    "type": contradiction_type,
                    "new_chunk_id": new_chunk.chunk_id,
                    "old_chunk_id": old_chunk.chunk_id,
                    "similarity": round(sim, 4),
                    "shared_intent_id": intent_id,
                    "resolution": resolution,
                    "timestamp": time.time(),
                }
                contradictions.append(record)
                self._contradiction_log.append(record)

        return contradictions

    @property
    def contradiction_log(self) -> List[dict]:
        """Full history of detected contradictions."""
        return list(self._contradiction_log)

    # ------------------------------------------------------------------ #
    #  DETECTION                                                          #
    # ------------------------------------------------------------------ #

    def _detect_contradiction(self, text_a: str, text_b: str) -> Optional[str]:
        """
        Heuristic contradiction detection. Returns the type of contradiction
        or None if no conflict is detected.
        """
        # 1. Negation asymmetry: one text negates, the other doesn't
        neg_a = len(_NEGATION_RE.findall(text_a))
        neg_b = len(_NEGATION_RE.findall(text_b))
        if abs(neg_a - neg_b) >= 1 and (neg_a > 0 or neg_b > 0):
            return "negation_conflict"

        # 2. Temporal conflict: same topic but different dates/times/days
        times_a = set(m.lower() for m in _TEMPORAL_RE.findall(text_a))
        times_b = set(m.lower() for m in _TEMPORAL_RE.findall(text_b))
        if times_a and times_b and times_a != times_b:
            # Both mention temporal info but different values
            return "temporal_conflict"

        # 3. Numeric conflict: same topic but different numbers
        nums_a = set(_NUMERIC_RE.findall(text_a))
        nums_b = set(_NUMERIC_RE.findall(text_b))
        if nums_a and nums_b and nums_a != nums_b:
            return "numeric_conflict"

        return None

    # ------------------------------------------------------------------ #
    #  RESOLUTION                                                         #
    # ------------------------------------------------------------------ #

    def _resolve(
        self, new_chunk: Chunk, old_chunk: Chunk, contradiction_type: str
    ) -> dict:
        """Apply resolution strategy and return metadata."""
        if self.strategy == "reinforcement":
            return self._resolve_by_reinforcement(new_chunk, old_chunk)
        elif self.strategy == "coexist":
            return self._resolve_coexist(new_chunk, old_chunk)
        else:
            return self._resolve_by_recency(new_chunk, old_chunk)

    def _resolve_by_recency(self, new_chunk: Chunk, old_chunk: Chunk) -> dict:
        """Newer information wins. Older chunk gets noise penalty."""
        if new_chunk.created_at >= old_chunk.created_at:
            winner, loser = new_chunk, old_chunk
        else:
            winner, loser = old_chunk, new_chunk

        loser.noise_score = min(1.0, loser.noise_score + self.noise_penalty)
        return {
            "strategy": "recency",
            "winner_id": winner.chunk_id,
            "loser_id": loser.chunk_id,
            "action": f"noise_penalty +{self.noise_penalty} applied to {loser.chunk_id}",
        }

    def _resolve_by_reinforcement(self, new_chunk: Chunk, old_chunk: Chunk) -> dict:
        """Higher reinforcement count wins."""
        if new_chunk.reinforcement_count >= old_chunk.reinforcement_count:
            winner, loser = new_chunk, old_chunk
        else:
            winner, loser = old_chunk, new_chunk

        loser.noise_score = min(1.0, loser.noise_score + self.noise_penalty)
        return {
            "strategy": "reinforcement",
            "winner_id": winner.chunk_id,
            "loser_id": loser.chunk_id,
            "action": f"noise_penalty +{self.noise_penalty} applied to {loser.chunk_id}",
        }

    def _resolve_coexist(self, new_chunk: Chunk, old_chunk: Chunk) -> dict:
        """Both survive. Contradiction is logged but no penalty applied."""
        return {
            "strategy": "coexist",
            "winner_id": None,
            "loser_id": None,
            "action": "logged_only",
        }

    # ------------------------------------------------------------------ #
    #  OPTIONAL LLM VERIFICATION                                         #
    # ------------------------------------------------------------------ #

    def _llm_verify(self, text_a: str, text_b: str) -> bool:
        """
        Use LLM to verify if two texts truly contradict each other.
        Returns True if contradiction is confirmed.
        """
        if not self.llm_client:
            return False

        try:
            prompt = (
                "Do these two statements contradict each other? "
                "Answer ONLY 'yes' or 'no'.\n\n"
                f"Statement A: {text_a[:500]}\n"
                f"Statement B: {text_b[:500]}"
            )
            response = self.llm_client.chat.completions.create(
                model=self.llm_model or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5,
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip().lower()
            return answer.startswith("yes")
        except Exception:
            return False
