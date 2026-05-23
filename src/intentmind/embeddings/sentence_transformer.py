from __future__ import annotations

from typing import List
from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers yüklü değil. Kurulum: pip install sentence-transformers"
            ) from exc
        self.model_name = model_name
        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()
