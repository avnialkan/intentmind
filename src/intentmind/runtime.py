from __future__ import annotations

from pathlib import Path
from .store import MemoryStore
from .embeddings import BaseEmbedder, FakeEmbedder
from .engines import IntentEngine, EmotionEngine, RecallEngine, EnergyEngine
from .builders import PromptBuilder
from .persistence import JsonPersistence


class IntentmindMemory:
    """Tek public API: add/query/save/load/tick/graph_summary."""

    def __init__(
        self,
        embedder: BaseEmbedder | None = None,
        store: MemoryStore | None = None,
        metadata: dict | None = None,
        is_test: bool = False,
        index_type: str = "exact",
        core_extractor: str = "llm",
        model: str | None = None,
        extractor=None,
    ):
        if embedder is None:
            if is_test:
                from .embeddings import FakeEmbedder
                self.embedder = FakeEmbedder()
            else:
                try:
                    from .embeddings import SentenceTransformerEmbedder
                    self.embedder = SentenceTransformerEmbedder()
                except ImportError:
                    raise ImportError(
                        "SentenceTransformer bulunamadı. Production modda 'sentence-transformers' "
                        "kurmalısınız (pip install sentence-transformers). Test için is_test=True parametresini "
                        "veya custom bir embedder kullanın."
                    )
        else:
            self.embedder = embedder

        effective_core_extractor = core_extractor
        if is_test and extractor is None and core_extractor == "llm":
            effective_core_extractor = "deterministic"

        
        if model is None:
            import os
            model = os.environ.get("OPENAI_MODEL")
            if not model:
                import warnings
                warnings.warn("OPENAI_MODEL is not set in environment. Falling back to default but it is highly recommended to set it (e.g., OPENAI_MODEL=gpt-4o-mini).")
                model = "gpt-4o-mini" # Failsafe for runtime, but examples won't have it.

        self._store = store or MemoryStore(index_type=index_type)
        self.metadata = metadata or {"embedder": getattr(self.embedder, "name", self.embedder.__class__.__name__), "total_queries": 0}
        self._intent_engine = IntentEngine(
            self._store,
            self.embedder,
            core_extractor=effective_core_extractor,
            model=model,
            extractor=extractor,
        )
        self._emotion = EmotionEngine(self.embedder)
        self._recall = RecallEngine(self._store)
        self._energy = EnergyEngine(self._store)
        self._prompt = PromptBuilder()

    def add(self, text: str, source: str = "chat", chunk_id: str | None = None) -> str:
        chunk = self._intent_engine.ingest(text=text, source=source, chunk_id=chunk_id)
        return chunk.chunk_id

    def query(self, user_query: str) -> dict:
        import time as _time

        t0 = _time.perf_counter()
        query_emb = self.embedder.embed(user_query)
        t_embed = _time.perf_counter()

        emotional_state = self._emotion.detect(query_emb)
        t_emotion = _time.perf_counter()

        query_token_embeddings = self._intent_engine.extract_query_intents(user_query)
        t_extract = _time.perf_counter()

        query_intent_labels = [label for label, _embedding in query_token_embeddings]
        
        recall_result = self._recall.recall(
            query=user_query, 
            query_embedding=query_emb, 
            emotional_state=emotional_state,
            query_intent_labels=query_intent_labels,
            query_token_embeddings=query_token_embeddings,
            energy_engine=self._energy,
        )
        t_recall = _time.perf_counter()

        prompt = self._prompt.build(user_query=user_query, recall_result=recall_result, emotional_state=emotional_state)
        t_prompt = _time.perf_counter()

        self.metadata["total_queries"] = self.metadata.get("total_queries", 0) + 1

        total_ms = (t_prompt - t0) * 1000
        latency_breakdown = {
            "embed_query_ms": round((t_embed - t0) * 1000, 2),
            "emotion_ms": round((t_emotion - t_embed) * 1000, 2),
            "extractor_ms": round((t_extract - t_emotion) * 1000, 2),
            "recall_ms": round((t_recall - t_extract) * 1000, 2),
            "prompt_ms": round((t_prompt - t_recall) * 1000, 2),
            "total_ms": round(total_ms, 2),
        }

        return {
            "prompt": prompt,
            "emotion": emotional_state.to_dict(),
            "memories": {
                "direct": len(recall_result["direct_memories"]),
                "associated": len(recall_result["associated_memories"]),
                "weak_echo": len(recall_result["weak_echo_memories"]),
                "rejected": len(recall_result["rejected_memories"]),
                "items": [
                    {
                        "chunk_id": item["chunk"].chunk_id,
                        "text": item["chunk"].text,
                        "score": item["score"],
                        "layer": item["layer"],
                        "intent": item["intent"].label,
                        "intent_ids": item["chunk"].intent_ids,
                        "source": item["chunk"].source,
                        "path_strength": round(item.get("path_strength", 0.0), 3),
                        "called_by": item.get("called_by"),
                        "reason": item.get("reason"),
                        "path": item.get("path", []),
                        "from_seed": item.get("from_seed", False),
                        "edge": self._edge_context(item),
                    }
                    for item in recall_result["direct_memories"] + recall_result["associated_memories"] + recall_result["weak_echo_memories"]
                ]
            },
            "stats": self._store.stats(),
            "cognitive_field": recall_result.get("cognitive_field", {}),
            "trace": self._build_trace(recall_result),
            "latency_breakdown": latency_breakdown,
            "extracted_query_intents": query_intent_labels,
        }

    def save(self, path: str | Path) -> None:
        self.metadata["embedder"] = getattr(self.embedder, "name", self.embedder.__class__.__name__)
        JsonPersistence.save(self._store, path, self.metadata)

    @classmethod
    def load(cls, path: str | Path, embedder: BaseEmbedder | None = None, is_test: bool = False) -> "IntentmindMemory":
        store, metadata = JsonPersistence.load(path)
        return cls(embedder=embedder, store=store, metadata=metadata, is_test=is_test)

    def tick(self, hours: float = 1.0) -> dict:
        energy_stats = self._energy.tick(hours=hours)
        cons_stats = self.consolidate()
        return {**energy_stats, **cons_stats}
        
    def consolidate(self) -> dict:
        return self._intent_engine.consolidate_memory()
        
    def visualize(self, output_html: str = "memory_map.html") -> str:
        from .vis import GraphVisualizer
        visualizer = GraphVisualizer(self._store)
        return visualizer.visualize(output_html)

    def graph_summary(self) -> dict:
        intents = list(self._store.intents.values())
        edges = list(self._store.edges.values())
        return {
            "total_intents": len(intents),
            "total_edges": len(edges),
            "total_chunks": len(self._store.chunks),
            "intent_list": [
                {
                    "label": i.label,
                    "type": i.type,
                    "extraction_confidence": round(i.extraction_confidence, 3),
                    "energy": round(i.energy, 3),
                    "source_count": i.source_count,
                    "activation_count": i.activation_count,
                    "state": i.state,
                }
                for i in sorted(intents, key=lambda x: x.energy, reverse=True)
            ],
            "edge_list": [
                {
                    "source": self._store.intents[e.source_id].label,
                    "target": self._store.intents[e.target_id].label,
                    "type": e.edge_type,
                    "weight": round(e.weight, 3),
                    "confidence": round(e.confidence, 3),
                    "support_count": e.support_count,
                    "evidence_count": len(e.evidence_chunk_ids),
                    "evidence_chunk_ids": e.evidence_chunk_ids[-3:],
                    "energy": round(e.energy, 3),
                    "state": e.state,
                    "co_activation_count": e.co_activation_count,
                }
                for e in sorted(edges, key=lambda x: (x.confidence, x.weight), reverse=True)
                if e.source_id in self._store.intents and e.target_id in self._store.intents
            ],
        }

    def _build_trace(self, recall_result) -> list:
        trace = []
        accepted_items = recall_result["direct_memories"] + recall_result["associated_memories"] + recall_result["weak_echo_memories"]
        for item in accepted_items:
            trace_entry = {
                "decision": "accepted",
                "chunk_id": item["chunk"].chunk_id,
                "score": item["score"],
                "layer": item["layer"],
                "intent": item["intent"].label,
                "domain": getattr(item["intent"], "domain", "general"),
                "reason": item.get("reason", "unknown_activation"),
                "called_by": item.get("called_by"),
                "path": item.get("path", []),
                "from_seed": item.get("from_seed", False),
            }
            
            if "called_by" in item and item["called_by"]:
                trace_entry["from"] = item["called_by"]
                trace_entry["to"] = item["intent"].label
                trace_entry["relation_role"] = item.get("edge_type", "weak_echo")
                trace_entry["confidence"] = item.get("edge_confidence", 0.0)
                
                evidence = item.get("edge_evidence", [])
                if evidence:
                    trace_entry["evidence_memory"] = evidence[0]
                    
                # Supernode penalty check
                intent_id = item["intent"].intent_id
                if getattr(self, "_energy", None) and hasattr(self._energy, "last_supernode_penalties"):
                    penalty = self._energy.last_supernode_penalties.get(intent_id, 0.0)
                    if penalty > 0:
                        trace_entry["supernode_penalty"] = round(penalty, 3)

            trace.append(trace_entry)
            
        for item in recall_result["rejected_memories"][:8]:
            trace.append({
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "layer": item["layer"],
                "reason": item["reason"],
                "decision": "rejected",
            })
        return trace

    def _edge_context(self, item: dict) -> dict:
        edge_id = item.get("edge_id")
        if not edge_id:
            return {}

        edge = self._store.edges.get(edge_id)
        if edge:
            evidence = edge.evidence_chunk_ids[-3:]
            weight = edge.weight
            confidence = edge.confidence
            support = edge.support_count
            state = edge.state
            edge_type = edge.edge_type
        else:
            evidence = item.get("edge_evidence", [])
            weight = item.get("edge_weight")
            confidence = item.get("edge_confidence")
            support = item.get("edge_support")
            state = item.get("edge_state")
            edge_type = item.get("edge_type")

        return {
            "edge_id": edge_id,
            "type": edge_type,
            "weight": round(weight, 3) if isinstance(weight, (int, float)) else weight,
            "confidence": round(confidence, 3) if isinstance(confidence, (int, float)) else confidence,
            "support_count": support,
            "state": state,
            "evidence_chunk_ids": evidence,
        }
