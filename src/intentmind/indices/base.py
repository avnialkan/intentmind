from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from ..models import IntentNode

class BaseIntentIndex(ABC):
    """
    Base interface for all Intent Search Indices (Exact, FAISS, etc.)
    """

    @abstractmethod
    def add(self, intent: IntentNode) -> None:
        """Add a new intent to the index."""
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], threshold: float = 0.80) -> Optional[IntentNode]:
        """
        Find the most similar existing intent by embedding cosine similarity.
        Returns the intent if similarity >= threshold, else None.
        """
        pass
