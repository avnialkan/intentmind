from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List
import time


@dataclass
class ExtractedIntent:
    label: str
    type: str = "concept"
    domain: str = "general"
    role: str = "topic"
    confidence: float = 1.0
    source: str = "extractor"
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    summary: str
    embedding: List[float]
    intent_ids: List[str]
    source: str = "chat"
    created_at: float = field(default_factory=time.time)
    feedback_score: float = 0.0
    noise_score: float = 0.0
    reinforcement_count: int = 1
    memory_tier: str = "episodic"  # working | episodic | semantic | archived

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


@dataclass
class IntentCandidate:
    text: str
    normalized: str
    embedding: List[float]
    intent_type: str = "concept"
    domain: str = "general"
    role: str = "topic"
    extraction_confidence: float = 1.0
    quality_score: float = 0.0
    concreteness_score: float = 0.0
    generic_score: float = 0.0
    matched_intent_id: str | None = None
    decision: str = "pending"  # reuse | create_active | create_candidate | reject
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IntentNode:
    intent_id: str
    label: str
    type: str
    embedding: List[float]
    domain: str = "general"
    energy: float = 0.50
    stability: float = 0.50
    extraction_confidence: float = 1.0
    activation_count: int = 0
    source_count: int = 0
    hub_score: float = 0.0
    noise_score: float = 0.0
    state: str = "active"  # active | candidate | weak | archived
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IntentNode":
        return cls(**data)


@dataclass
class IntentEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    energy: float = 0.50
    confidence: float = 0.50
    support_count: int = 1
    evidence_chunk_ids: List[str] = field(default_factory=list)
    co_activation_count: int = 0
    state: str = "active"  # active | weak | candidate | archived
    watch_count: int = 0
    last_reinforced: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IntentEdge":
        return cls(**data)


@dataclass
class CognitiveState:
    current: str = "neutral"
    confidence: float = 1.0
    weak_echo: bool = False

    def to_dict(self) -> dict:
        return asdict(self)
