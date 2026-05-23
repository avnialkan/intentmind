from __future__ import annotations

import math
import random
from typing import List
from .base import BaseEmbedder

SEMANTIC_GROUPS = {
    "memory":    ["hafıza", "memory", "recall", "chunk", "store", "geçmiş"],
    "graph":     ["graph", "intent", "edge", "node", "traversal", "layer", "bağ"],
    "energy":    ["energy", "enerji", "decay", "zayıflama", "güçlenme", "stability"],
    "emotion":   ["emotion", "duygu", "emotional", "merak", "heyecan", "şüphe", "korku"],
    "score":     ["score", "activation", "similarity", "threshold", "weight", "ağırlık"],
    "prompt":    ["prompt", "builder", "context", "bağlam", "llm", "cevap"],
    "pruning":   ["pruning", "budama", "temizleme", "pollution", "explosion", "duplicate"],
    "car":       ["car", "araba", "arabayla", "arabanın", "arabanin", "arabam", "arabayi", "servis", "servise"],
    "insurance": ["insurance", "sigorta", "sigortasini"],
    "money":     ["money", "para", "param"],
    "travel":    ["london", "antalya", "antalyaya", "yolculuk", "istanbul", "istanbula", "goturecegim", "gitmem", "yarin", "gidecegim"],
    "fuel":      ["benzin", "benzini", "bitti"],
    "food":      ["food", "meal", "yemek", "yemistik"],
}

DIM = 32


def _group_vector(group_idx: int, dim: int = DIM) -> List[float]:
    vec = [0.0] * dim
    width = max(1, dim // len(SEMANTIC_GROUPS))
    start = group_idx * width
    end = start + width
    for i in range(start, min(end, dim)):
        vec[i] = 1.0
    return vec


GROUP_VECTORS = {group: _group_vector(idx) for idx, group in enumerate(SEMANTIC_GROUPS)}


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class FakeEmbedder(BaseEmbedder):
    name = "fake-semantic-cluster"

    def __init__(self, dim: int = DIM):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        text_l = text.lower()
        group_weights = {}
        for group, keywords in SEMANTIC_GROUPS.items():
            hits = sum(1 for kw in keywords if kw in text_l)
            if hits > 0:
                group_weights[group] = hits

        import hashlib
        seed_int = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        random.seed(seed_int)
        vec = [random.gauss(0, 0.15) for _ in range(self.dim)]

        if group_weights:
            total = sum(group_weights.values())
            for group, w in group_weights.items():
                gv = GROUP_VECTORS[group]
                alpha = (w / total) * 2.0
                for i in range(min(self.dim, len(gv))):
                    vec[i] += alpha * gv[i]

        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return [1.0 / math.sqrt(self.dim)] * self.dim
        return [x / norm for x in vec]
