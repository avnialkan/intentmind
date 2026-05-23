import time
from types import SimpleNamespace
from ..embeddings import cosine_similarity

class CognitiveField:
    """
    CognitiveField represents the energy diffusion graph of a query.
    It isolates the responsibility of computing intent activations and layer propagations
    from the RecallEngine, which should only be responsible for chunk selection.
    """
    def __init__(self, store, query_embedding, cognitive_state, query_intent_label_set, query_match_intent_ids, field_seed_ids, energy_engine=None):
        self.store = store
        self.query_embedding = query_embedding
        self.cognitive_state = cognitive_state
        self.query_intent_label_set = set(query_intent_label_set or [])
        self.query_match_intent_ids = set(query_match_intent_ids or [])
        self.field_seed_ids = set(field_seed_ids or [])
        self.energy_engine = energy_engine
        
        self.layers = {0: [], 1: [], 2: [], 3: []}
        self.seen_intents = set()
        self.current_time = time.time()

    def activation_score(self, intent):
        query_similarity = cosine_similarity(self.query_embedding, intent.embedding)
        edges = self.store.get_neighbors(intent.intent_id)
        edge_energy = sum(e.energy for _, e in edges) / len(edges) if edges else 0.40
        seconds_since = self.current_time - intent.last_active
        recency_boost = max(0.0, 1.0 - (seconds_since / 3600))
        
        # Calculate base activation without fixed project_relevance
        base = (
            query_similarity * 0.45
            + intent.energy * 0.20
            + edge_energy * 0.15
            + recency_boost * 0.20
            - intent.hub_score * 0.15
            - getattr(intent, "noise_score", 0.0) * 0.20
        )
        
        multipliers = {
            "neutral": 1.00, "curious": 1.00,
            "confident": 1.15, "skeptical": 1.10, "urgent": 1.25, "cautious": 1.20,
        }
        m = multipliers.get(self.cognitive_state.current, 1.00)
        factor = 1.0 + ((m - 1.0) * self.cognitive_state.confidence)
        return round(base * factor, 4)

    def _edge_path_score(self, source_intent, target_intent, edge, multiplier: float = 1.0) -> float:
        confidence_factor = 0.75 + min(1.0, edge.confidence) * 0.25
        support_factor = 1.0 + min(0.12, max(0, edge.support_count - 1) * 0.04)
        
        # Universal Jump Brake Multipliers
        role_multipliers = {
            "weak_echo": 0.10, # Increased from 0.0
            "co_occurrence_link": 0.75,
            "thematic_link": 0.3,
            "spatial_link": 0.6,
            "actor_link": 0.6,
            "event_link": 0.9,
            "causal_link": 0.9,
            "instrumental_link": 0.9,
        }
        edge_multiplier = role_multipliers.get(edge.edge_type, 0.5)
        
        # Cross-Domain Penalty
        domain_penalty = 1.0
        if getattr(source_intent, "domain", "general") != getattr(target_intent, "domain", "general"):
            domain_penalty = 0.15
            
        return round(edge.weight * confidence_factor * support_factor * multiplier * edge_multiplier * domain_penalty, 4)

    def _dedupe_layer(self, items):
        best = {}
        for item in items:
            intent = item["intent"]
            if intent.intent_id not in best or item["score"] > best[intent.intent_id]["score"]:
                best[intent.intent_id] = item
        return sorted(best.values(), key=lambda x: x["score"], reverse=True)

    def propagate(self, ticks=3):
        # Layer 0 (Core Activation)
        for intent in self.store.intents.values():
            if intent.state == "archived":
                continue
            
            score = self.activation_score(intent)
            is_embedding_match = intent.intent_id in self.query_match_intent_ids
            is_label_match = intent.label in self.query_intent_label_set

            # Partial label match
            if not is_label_match:
                intent_words = set(intent.label.lower().split())
                for ql in self.query_intent_label_set:
                    ql_words = set(ql.lower().split())
                    if ql_words and intent_words and (ql_words.issubset(intent_words) or intent_words.issubset(ql_words)):
                        is_label_match = True
                        break

            is_direct = is_embedding_match or is_label_match

            # Conversational Priming
            is_primed = False
            if self.energy_engine and not is_direct:
                current_energy = self.energy_engine.get_energy(intent, self.current_time)
                if current_energy >= 0.65:
                    is_primed = True

            if intent.state != "active" and not (is_direct or is_primed):
                continue

            query_sim = cosine_similarity(self.query_embedding, intent.embedding)
            
            if (score >= 0.65 and query_sim >= 0.50) or is_direct or is_primed:
                final_score = max(score, 0.75) if is_direct else score
                final_score = max(final_score, 0.65) if is_primed else final_score
                reason = "direct_match" if is_direct else ("conversational_priming" if is_primed else "vector_similarity")
                
                self.layers[0].append({
                    "intent": intent,
                    "score": final_score,
                    "path_strength": final_score,
                    "called_by": None,
                    "reason": reason,
                    "path": [intent.label],
                    "from_seed": intent.intent_id in self.field_seed_ids,
                })
                self.seen_intents.add(intent.intent_id)

        # Seed Layer 0 with intents from the most semantically relevant chunks
        top_chunks = self.store.search_chunks(self.query_embedding, top_k=3)
        for chunk_score, chunk in top_chunks:
            if chunk_score > 0.60:
                chunk_intents = []
                for intent_id in chunk.intent_ids:
                    if intent_id in self.seen_intents:
                        continue
                    intent = self.store.intents.get(intent_id)
                    if not intent or intent.state not in ["active", "candidate"]:
                        continue
                    intent_query_similarity = cosine_similarity(self.query_embedding, intent.embedding)
                    is_matched = intent.intent_id in self.query_match_intent_ids or intent.label in self.query_intent_label_set
                    if not is_matched and intent_query_similarity < 0.50:
                        continue
                    chunk_intents.append((intent_query_similarity, intent))

                for intent_query_similarity, intent in sorted(chunk_intents, reverse=True, key=lambda x: x[0])[:2]:
                    self.layers[0].append({
                        "intent": intent,
                        "score": max(chunk_score, intent_query_similarity),
                        "path_strength": max(chunk_score, intent_query_similarity),
                        "called_by": None,
                        "reason": "contextual_activation",
                        "path": [intent.label],
                        "from_seed": intent.intent_id in self.field_seed_ids,
                    })
                    self.seen_intents.add(intent.intent_id)

        self.layers[0] = sorted(self.layers[0], key=lambda x: x["score"], reverse=True)[:12]
        seen_l0 = {item["intent"].intent_id for item in self.layers[0]}
        
        # Layer 1
        for item in self.layers[0]:
            intent = item["intent"]
            for path_order, (neighbor_id, edge) in enumerate(self.store.get_neighbors(intent.intent_id)):
                neighbor = self.store.intents.get(neighbor_id)
                if neighbor_id in seen_l0 and neighbor and neighbor.label in self.query_intent_label_set:
                    continue

                if edge.edge_type == "co_occurrence_link":
                    if neighbor:
                        intent_sim = cosine_similarity(intent.embedding, neighbor.embedding)
                        if intent_sim < 0.40:
                            continue

                if neighbor and neighbor.state == "active" and edge.energy >= 0.45 and edge.weight >= 0.40 and edge.confidence >= 0.50:
                    edge_score = self._edge_path_score(intent, neighbor, edge)
                    self.layers[1].append({
                        "intent": neighbor, 
                        "score": edge_score, 
                        "path_strength": edge_score,
                        "called_by": intent.label,
                        "reason": "neighbor_intent_reactivation",
                        "path": item["path"] + [neighbor.label],
                        "edge_id": edge.edge_id,
                        "edge_type": edge.edge_type,
                        "edge_weight": edge.weight,
                        "edge_confidence": edge.confidence,
                        "edge_support": edge.support_count,
                        "edge_state": edge.state,
                        "edge_evidence": edge.evidence_chunk_ids[-3:],
                        "path_order": path_order,
                        "from_seed": item.get("from_seed", False),
                    })
        self.layers[1] = self._dedupe_layer(self.layers[1])[:12]

        if self.cognitive_state.weak_echo and ticks >= 2:
            seen_l1 = {item["intent"].intent_id for item in self.layers[1]}
            self.seen_intents = seen_l0 | seen_l1
            
            # Layer 2
            for item in self.layers[1]:
                if item.get("edge_type") == "weak_echo":
                    continue
                intent = item["intent"]
                for path_order, (neighbor_id, edge) in enumerate(self.store.get_neighbors(intent.intent_id)):
                    if neighbor_id in self.seen_intents:
                        continue
                    neighbor = self.store.intents.get(neighbor_id)
                    if neighbor and neighbor.state == "active" and edge.energy >= 0.35 and edge.weight >= 0.35 and edge.confidence >= 0.45:
                        edge_score = self._edge_path_score(intent, neighbor, edge, multiplier=0.85)
                        self.layers[2].append({
                            "intent": neighbor, 
                            "score": edge_score, 
                            "path_strength": edge_score,
                            "called_by": intent.label,
                            "reason": "weak_echo_reactivation",
                            "path": item["path"] + [neighbor.label],
                            "edge_id": edge.edge_id,
                            "edge_type": edge.edge_type,
                            "edge_weight": edge.weight,
                            "edge_confidence": edge.confidence,
                            "edge_support": edge.support_count,
                            "edge_state": edge.state,
                            "edge_evidence": edge.evidence_chunk_ids[-3:],
                            "path_order": item.get("path_order", 0) * 100 + path_order,
                            "from_seed": item.get("from_seed", False),
                        })
            self.layers[2] = self._dedupe_layer(self.layers[2])[:8]

            seen_l2 = {item["intent"].intent_id for item in self.layers[2]}
            self.seen_intents = self.seen_intents | seen_l2
            
            # Layer 3
            if ticks >= 3:
                for item in self.layers[2]:
                    if item.get("edge_type") == "weak_echo":
                        continue
                    intent = item["intent"]
                    for path_order, (neighbor_id, edge) in enumerate(self.store.get_neighbors(intent.intent_id)):
                        if neighbor_id in self.seen_intents:
                            continue
                        neighbor = self.store.intents.get(neighbor_id)
                        if neighbor and neighbor.state == "active" and edge.energy >= 0.25 and edge.weight >= 0.25 and edge.confidence >= 0.40:
                            edge_score = self._edge_path_score(intent, neighbor, edge, multiplier=0.65)
                            self.layers[3].append({
                                "intent": neighbor, 
                                "score": edge_score, 
                                "path_strength": edge_score,
                                "called_by": intent.label,
                                "reason": "faint_echo_reactivation",
                                "path": item["path"] + [neighbor.label],
                                "edge_id": edge.edge_id,
                                "edge_type": edge.edge_type,
                                "edge_weight": edge.weight,
                                "edge_confidence": edge.confidence,
                                "edge_support": edge.support_count,
                                "edge_state": edge.state,
                                "edge_evidence": edge.evidence_chunk_ids[-3:],
                                "path_order": item.get("path_order", 0) * 100 + path_order,
                                "from_seed": item.get("from_seed", False),
                            })
                self.layers[3] = self._dedupe_layer(self.layers[3])[:4]
                
    def to_layers(self):
        return self.layers
