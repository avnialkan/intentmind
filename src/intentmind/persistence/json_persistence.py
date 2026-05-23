from __future__ import annotations

import json
import time
from pathlib import Path
from ..models import Chunk, IntentNode, IntentEdge
from ..store import MemoryStore


class JsonPersistence:
    VERSION = "0.1"

    @staticmethod
    def save(store: MemoryStore, path: str | Path, metadata: dict | None = None) -> None:
        data = {
            "version": JsonPersistence.VERSION,
            "created_at": time.time(),
            "metadata": metadata or {},
            "chunks": [chunk.to_dict() for chunk in store.chunks.values()],
            "intents": [intent.to_dict() for intent in store.intents.values()],
            "edges": [edge.to_dict() for edge in store.edges.values()],
        }
        path = Path(path)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> tuple[MemoryStore, dict]:
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        store = MemoryStore()
        for raw in data.get("chunks", []):
            chunk = Chunk.from_dict(raw)
            store.chunks[chunk.chunk_id] = chunk
        for raw in data.get("intents", []):
            intent = IntentNode.from_dict(raw)
            store.intents[intent.intent_id] = intent
            store.index.add(intent)
        for raw in data.get("edges", []):
            edge = IntentEdge.from_dict(raw)
            store.edges[edge.edge_id] = edge
        return store, data.get("metadata", {})
