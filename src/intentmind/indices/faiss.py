from typing import List, Optional
from .base import BaseIntentIndex
from ..models import IntentNode

try:
    import numpy as np
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    np = None
    faiss = None
    FAISS_AVAILABLE = False


class FaissIntentIndex(BaseIntentIndex):
    """
    FAISS-backed HNSW/Flat index for scalable intent search.
    Requires 'faiss-cpu' to be installed.
    """
    def __init__(self, embedding_dim: int = 384):
        if not FAISS_AVAILABLE:
            raise ImportError("faiss-cpu is not installed. Please install it with 'pip install faiss-cpu' or fallback to exact index.")
        
        self.embedding_dim = embedding_dim
        # IndexFlatIP uses Inner Product which is equivalent to Cosine Similarity for normalized vectors
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        
        self.intents: List[IntentNode] = []

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec, axis=-1, keepdims=True)
        # Avoid division by zero
        norm = np.where(norm == 0, 1e-10, norm)
        return vec / norm

    def add(self, intent: IntentNode) -> None:
        self.intents.append(intent)
        
        # Add to FAISS index
        vec = np.array([intent.embedding], dtype=np.float32)
        vec = self._normalize(vec)
        self.index.add(vec)

    def search(self, query_embedding: List[float], threshold: float = 0.80) -> Optional[IntentNode]:
        if self.index.ntotal == 0:
            return None
            
        vec = np.array([query_embedding], dtype=np.float32)
        vec = self._normalize(vec)
        
        # Search top 10 to skip archived ones
        k = min(10, self.index.ntotal)
        distances, indices = self.index.search(vec, k)
        
        best_intent = None
        best_sim = -1.0
        
        for sim, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
                
            intent = self.intents[idx]
            if intent.state == "archived":
                continue
                
            if sim > best_sim:
                best_sim = float(sim)
                best_intent = intent
                
            # Since FAISS returns them sorted by similarity, the first non-archived is the best
            break
            
        if best_intent and best_sim >= threshold:
            return best_intent
        return None
