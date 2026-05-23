from typing import List, Optional
from .base import BaseIntentIndex
from ..models import IntentNode
from ..embeddings import cosine_similarity

class ExactIntentIndex(BaseIntentIndex):
    """
    Standard exact search loop index.
    Iterates through all intents and calculates cosine similarity.
    Ideal for < 10,000 nodes.
    """
    def __init__(self):
        self.intents: List[IntentNode] = []

    def add(self, intent: IntentNode) -> None:
        self.intents.append(intent)

    def search(self, query_embedding: List[float], threshold: float = 0.80) -> Optional[IntentNode]:
        best_intent = None
        best_sim = -1.0
        for intent in self.intents:
            if intent.state == "archived":
                continue
            sim = cosine_similarity(query_embedding, intent.embedding)
            if sim > best_sim:
                best_sim = sim
                best_intent = intent
        
        if best_intent and best_sim >= threshold:
            return best_intent
        return None
