from __future__ import annotations

import uuid
from typing import Dict, List, Tuple, Optional
from .models import Chunk, IntentNode, IntentEdge
from .embeddings import cosine_similarity


from .indices import ExactIntentIndex, FaissIntentIndex

class MemoryStore:
    def __init__(self, index_type: str = "exact"):
        self.chunks: Dict[str, Chunk] = {}
        self.intents: Dict[str, IntentNode] = {}
        self.edges: Dict[str, IntentEdge] = {}
        self._neighbor_cache: Dict[str, List[Tuple[str, IntentEdge]]] | None = None
        
        self.index_type = index_type.lower()
        if self.index_type == "faiss":
            try:
                self.index = FaissIntentIndex()
            except ImportError:
                print("[Warning] FAISS not available. Falling back to ExactIntentIndex.")
                self.index_type = "exact"
                self.index = ExactIntentIndex()
        else:
            self.index = ExactIntentIndex()

    def add_chunk(self, text: str, summary: str, embedding: List[float], intent_ids: List[str], source: str = "chat", chunk_id: str | None = None) -> Chunk:
        chunk_id = chunk_id or f"chunk_{uuid.uuid4().hex[:8]}"
        if chunk_id in self.chunks:
            raise ValueError(f"Chunk ID '{chunk_id}' already exists in store.")
        chunk = Chunk(chunk_id=chunk_id, text=text, summary=summary, embedding=embedding, intent_ids=intent_ids, source=source)
        self.chunks[chunk_id] = chunk
        return chunk

    def add_intent(
        self,
        label: str,
        intent_type: str,
        embedding: List[float],
        extraction_confidence: float = 1.0,
    ) -> IntentNode:
        intent_id = f"intent_{uuid.uuid4().hex[:8]}"
        intent = IntentNode(
            intent_id=intent_id,
            label=label,
            type=intent_type,
            embedding=embedding,
            extraction_confidence=extraction_confidence,
        )
        self.intents[intent_id] = intent
        self.index.add(intent)  # Add to search index
        return intent

    def find_intent_by_label(self, label: str):
        normalized = label.strip().lower()
        for intent in self.intents.values():
            labels_to_check = [intent.label.strip().lower()]
            if intent.lemma:
                labels_to_check.append(intent.lemma.strip().lower())
            labels_to_check.extend([a.strip().lower() for a in intent.aliases])
            
            if normalized in labels_to_check:
                return intent
        return None

    def find_intent_by_embedding(self, embedding: List[float], threshold: float = 0.80) -> Optional[IntentNode]:
        """
        Find the most similar existing intent by embedding cosine similarity.
        Returns the intent if similarity >= threshold, else None.
        Delegates search to the underlying index (exact or faiss).
        """
        return self.index.search(embedding, threshold)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float,
        state: str | None = None,
        evidence_chunk_id: str | None = None,
    ) -> IntentEdge:
        if source_id == target_id:
            raise ValueError("self-edge oluşturulamaz")
        for edge in self.edges.values():
            same = (edge.source_id == source_id and edge.target_id == target_id)
            reverse = (edge.source_id == target_id and edge.target_id == source_id)
            if same or reverse:
                was_new_evidence = self._add_edge_evidence(edge, evidence_chunk_id)
                if was_new_evidence:
                    edge.support_count += 1
                edge.co_activation_count += 1
                edge.weight = min(1.0, max(edge.weight, weight) + 0.02)
                edge.energy = min(1.0, edge.energy + 0.03)
                edge.confidence = self._edge_confidence(edge)
                edge.state = state or self._edge_state(edge.confidence, edge.support_count)
                self._invalidate_neighbor_cache()
                return edge
        edge_id = f"edge_{uuid.uuid4().hex[:8]}"
        evidence = [evidence_chunk_id] if evidence_chunk_id else []
        support_count = max(1, len(evidence))
        edge = IntentEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            confidence=0.0, # calculated below
            support_count=support_count,
            evidence_chunk_ids=evidence,
            state="candidate" # defaults to candidate
        )
        edge.confidence = self._edge_confidence(edge)
        edge.state = state or self._edge_state(edge.confidence, edge.support_count)
        self.edges[edge_id] = edge
        self._invalidate_neighbor_cache()
        return edge

    def _invalidate_neighbor_cache(self) -> None:
        self._neighbor_cache = None

    def _add_edge_evidence(self, edge: IntentEdge, chunk_id: str | None) -> bool:
        if not chunk_id or chunk_id in edge.evidence_chunk_ids:
            return False
        edge.evidence_chunk_ids.append(chunk_id)
        if len(edge.evidence_chunk_ids) > 20:
            edge.evidence_chunk_ids = edge.evidence_chunk_ids[-20:]
        return True

    def _edge_confidence(self, edge: IntentEdge) -> float:
        # co_occurrence_frequency
        freq_factor = min(1.0, 0.3 + (edge.support_count * 0.15))
        
        # semantic_consistency
        sem_factor = max(0.1, edge.weight)
        
        # temporal recurrence
        temp_factor = max(0.1, edge.energy)
        
        # domain alignment
        domain_factor = 1.0
        source = self.intents.get(edge.source_id)
        target = self.intents.get(edge.target_id)
        if source and target and source.domain != "general" and target.domain != "general":
            if source.domain != target.domain:
                domain_factor = 0.6
                
        base_confidence = freq_factor * sem_factor * temp_factor * domain_factor
        return round(min(1.0, base_confidence), 4)

    def _edge_state(self, confidence: float, support_count: int) -> str:
        # Friend's feedback: "edge hemen oluşmamalı. ilk karşılaşma: candidate edge."
        if support_count <= 1:
            return "candidate"
        if confidence >= 0.65 and support_count >= 3:
            return "active"
        if confidence >= 0.40 and support_count >= 2:
            return "weak"
        return "candidate"

    def get_chunks_by_intent(self, intent_id: str) -> List[Chunk]:
        return [chunk for chunk in self.chunks.values() if intent_id in chunk.intent_ids]

    def search_chunks(self, embedding: List[float], top_k: int = 5) -> List[Tuple[float, Chunk]]:
        """
        Global semantic search over chunks (Fallback to classic vector RAG).
        """
        results = []
        for chunk in self.chunks.values():
            sim = cosine_similarity(embedding, chunk.embedding)
            results.append((sim, chunk))
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def get_neighbors(self, intent_id: str) -> List[Tuple[str, IntentEdge]]:
        if self._neighbor_cache is None:
            cache: Dict[str, List[Tuple[str, IntentEdge]]] = {}
            for edge in self.edges.values():
                cache.setdefault(edge.source_id, []).append((edge.target_id, edge))
                cache.setdefault(edge.target_id, []).append((edge.source_id, edge))
            self._neighbor_cache = cache

        neighbors = [
            (neighbor_id, edge)
            for neighbor_id, edge in self._neighbor_cache.get(intent_id, [])
            if edge.state in {"active", "weak"}
        ]
        return sorted(neighbors, key=lambda item: (item[1].confidence, item[1].weight), reverse=True)

    def stats(self) -> dict:
        return {"chunks": len(self.chunks), "intents": len(self.intents), "edges": len(self.edges)}
