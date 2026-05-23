from .base import BaseEmbedder
from .fake import FakeEmbedder, cosine_similarity
try:
    from .sentence_transformer import SentenceTransformerEmbedder
except Exception:  # optional dependency
    SentenceTransformerEmbedder = None

__all__ = ["BaseEmbedder", "FakeEmbedder", "SentenceTransformerEmbedder", "cosine_similarity"]
