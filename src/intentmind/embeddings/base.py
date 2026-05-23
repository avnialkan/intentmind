from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    name: str = "base"

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        raise NotImplementedError
