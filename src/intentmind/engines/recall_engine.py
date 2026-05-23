from __future__ import annotations
from .cognitive_field import CognitiveField

import time
from types import SimpleNamespace
from ..store import MemoryStore
from ..embeddings import cosine_similarity


class RecallEngine:
    def __init__(self, store: MemoryStore, intent_match_threshold: float = 0.68):
        self.store = store
        self.intent_match_threshold = intent_match_threshold



    def recall(
        self,
        query,
        query_embedding,
        cognitive_state,
        query_intent_labels=None,
        query_token_embeddings=None,
        energy_engine=None,
    ):
        query_intent_labels = query_intent_labels or []
        query_intent_set = set(query_intent_labels)
        query_token_embeddings = query_token_embeddings or []
        query_match_intent_ids = self._matching_intent_ids(query_intent_labels, query_token_embeddings)
    # Memory tier multipliers for recall scoring
    _TIER_MULTIPLIERS = {
        "working": 1.20,    # Fresh context — boosted
        "episodic": 1.00,   # Baseline
        "semantic": 1.10,   # Consolidated knowledge — trusted
        "archived": 0.50,   # Fading — heavily penalized
    }

    def chunk_score(self, query_embedding, chunk, intent, path_strength, layer_id=0):
        query_chunk_sim = cosine_similarity(query_embedding, chunk.embedding)
        seconds_since = time.time() - chunk.created_at
        novelty_score = max(0.0, 1.0 - (seconds_since / 3600))
        intent_match = path_strength # use path strength as intent relevance
        
        # Memory tier multiplier (importance)
        tier_mult = self._TIER_MULTIPLIERS.get(
            getattr(chunk, "memory_tier", "episodic"), 1.00
        )
        
        # Trust/Reliability factor
        trust = 1.0 + min(0.5, chunk.reinforcement_count * 0.05) + (chunk.feedback_score * 0.1)
        if getattr(chunk, "noise_score", 0) > 0.5:
            trust *= 0.5
            
        if layer_id > 0:
            intent_chunk_sim = cosine_similarity(intent.embedding, chunk.embedding)
            relevance = max(0.1, path_strength) * max(0.1, intent_chunk_sim)
            relevance += max(0, query_chunk_sim) * 0.20
        else:
            relevance = max(0.1, query_chunk_sim) * max(0.1, intent_match)
            
        base = (
            relevance * 
            max(0.5, novelty_score) * 
            tier_mult * 
            trust
        )
        return round(base, 4)

    def recall(
        self,
        query,
        query_embedding,
        cognitive_state,
        query_intent_labels=None,
        query_token_embeddings=None,
        energy_engine=None,
    ):
        query_intent_labels = query_intent_labels or []
        query_intent_set = set(query_intent_labels)
        query_token_embeddings = query_token_embeddings or []
        query_match_intent_ids = self._matching_intent_ids(query_intent_labels, query_token_embeddings)
        
        field = CognitiveField(
            self.store,
            query_embedding,
            cognitive_state,
            query_intent_labels,
            query_match_intent_ids,
            [],
            energy_engine
        )
        field.propagate(ticks=3)
        layers = field.to_layers()
        
        cognitive_field = {"seed_intents": [], "activated_intents": [], "gates": {}}
        field_seed_ids = []
        if energy_engine is not None:
            field_seed_ids = self._field_seed_ids(layers)
            if field_seed_ids:
                cognitive_field = energy_engine.activate_field(field_seed_ids)
                field = CognitiveField(
                    self.store,
                    query_embedding,
                    cognitive_state,
                    query_intent_labels,
                    query_match_intent_ids,
                    field_seed_ids,
                    energy_engine
                )
                field.propagate(ticks=3)
                layers = field.to_layers()

        thresholds = {0: 0.25, 1: 0.38, 2: 0.35, 3: 0.45}
        candidates, rejected = [], []
        
        for layer_id, intent_items in layers.items():
            for item in intent_items:
                intent = item["intent"]
                path_strength = item["path_strength"]
                called_by = item.get("called_by")
                
                for chunk in self.store.get_chunks_by_intent(intent.intent_id):
                    score = self.chunk_score(query_embedding, chunk, intent, path_strength, layer_id=layer_id)
                    query_intent_overlap = self._query_intent_overlap(
                        chunk,
                        query_intent_set,
                        query_token_embeddings,
                        query_match_intent_ids,
                    )
                    query_exact_overlap = self._query_exact_overlap(chunk, query_intent_set)
                    query_chunk_similarity = cosine_similarity(query_embedding, chunk.embedding)
                    intent_query_similarity = cosine_similarity(query_embedding, intent.embedding)
                    intent_chunk_similarity = cosine_similarity(intent.embedding, chunk.embedding)
                    candidate_label_exact = intent.label in query_intent_set
                    candidate_intent_matched = intent.intent_id in query_match_intent_ids or candidate_label_exact
                    edge_support = item.get("edge_support", 0) or 0
                    edge_confidence = item.get("edge_confidence", 0.0) or 0.0

                    if query_intent_overlap and (candidate_intent_matched or intent_chunk_similarity >= 0.30):
                        score += min(0.18, 0.06 * query_intent_overlap)

                    if layer_id == 0 and item["reason"] == "direct_match" and query_intent_overlap:
                        score += min(0.12, 0.06 * query_intent_overlap)

                    if layer_id > 0:
                        if intent_chunk_similarity < 0.25:
                            score -= 0.18
                        if query_intent_overlap == 0 and edge_support <= 1:
                            score -= 0.22
                        elif query_intent_overlap == 1 and edge_support <= 1 and edge_confidence < 0.56:
                            score -= 0.08

                        # Semantic relevance gate: HARD REJECT chunks whose
                        # actual text is unrelated to the query. No mercy.
                        # EXCEPT for chat chunks — conversational context relies on the graph, not text matching.
                        if query_chunk_similarity < 0.30 and chunk.source != "chat":
                            continue  # Skip entirely — this chunk is noise
                    
                    # Association bonus
                    if called_by:
                        called_by_intent = self.store.find_intent_by_label(called_by)
                        if called_by_intent and called_by_intent.intent_id in chunk.intent_ids:
                            score += 0.05
                            
                    if score >= thresholds[layer_id]:
                        cand = {
                            "chunk": chunk, 
                            "score": round(score, 4), 
                            "layer": layer_id, 
                            "intent": intent, 
                            "path_strength": path_strength,
                            "called_by": called_by,
                            "reason": item["reason"],
                            "path": item["path"],
                            "path_order": item.get("path_order", 0),
                            "query_intent_overlap": query_intent_overlap,
                            "query_exact_overlap": query_exact_overlap,
                            "query_intent_count": len(query_token_embeddings),
                            "query_chunk_similarity": round(query_chunk_similarity, 4),
                            "intent_query_similarity": round(intent_query_similarity, 4),
                            "intent_chunk_similarity": round(intent_chunk_similarity, 4),
                            "candidate_label_exact": candidate_label_exact,
                            "candidate_intent_matched": candidate_intent_matched,
                            "from_seed": item.get("from_seed", False),
                            "called_by_exact": called_by in query_intent_set if called_by else False,
                            "edge_id": item.get("edge_id"),
                            "edge_type": item.get("edge_type"),
                            "edge_confidence": item.get("edge_confidence"),
                            "edge_support": item.get("edge_support"),
                            "edge_state": item.get("edge_state"),
                            "edge_evidence": item.get("edge_evidence", []),
                        }
                        candidates.append(cand)
                    else:
                        rejected.append({"chunk_id": chunk.chunk_id, "score": score, "layer": layer_id, "reason": "below_threshold"})

        candidates.extend(
            self._semantic_chunk_candidates(
                query_embedding,
                query_intent_set,
                query_token_embeddings,
                query_match_intent_ids,
            )
        )
        candidates.extend(
            self._llm_intent_chunk_candidates(
                query_embedding,
                query_intent_set,
                query_token_embeddings,
                query_match_intent_ids,
            )
        )

        candidates = sorted(self.remove_duplicates(candidates), key=lambda x: x["score"], reverse=True)
        direct = self._select_direct(candidates)
        associated = self._select_associated(candidates, direct, query_intent_set)
        weak_echo = [c for c in candidates if c["layer"] == 3][:1] if cognitive_state.weak_echo else []
        
        return {
            "direct_memories": direct,
            "associated_memories": associated,
            "weak_echo_memories": weak_echo,
            "rejected_memories": rejected,
            "activated_layers": layers,
            "cognitive_field": cognitive_field,
        }

    def _semantic_chunk_candidates(
        self,
        query_embedding,
        query_intent_set,
        query_token_embeddings,
        query_match_intent_ids,
        top_k: int = 5,
    ):
        candidates = []
        for query_chunk_similarity, chunk in self.store.search_chunks(query_embedding, top_k=top_k):
            if query_chunk_similarity < 0.35:
                continue

            best = self._best_chunk_intent(query_embedding, chunk)
            if not best:
                continue
            intent, intent_query_similarity, intent_chunk_similarity = best
            query_intent_overlap = self._query_intent_overlap(
                chunk,
                query_intent_set,
                query_token_embeddings,
                query_match_intent_ids,
            )
            query_exact_overlap = self._query_exact_overlap(chunk, query_intent_set)
            candidate_label_exact = intent.label in query_intent_set
            candidate_intent_matched = intent.intent_id in set(query_match_intent_ids or []) or candidate_label_exact

            score = (
                query_chunk_similarity * 0.55
                + max(intent_query_similarity, 0.0) * 0.25
                + max(intent_chunk_similarity, 0.0) * 0.15
                + min(0.05, chunk.reinforcement_count * 0.01)
            )

            candidates.append({
                "chunk": chunk,
                "score": round(score, 4),
                "layer": 0,
                "intent": intent,
                "path_strength": round(max(query_chunk_similarity, intent_query_similarity), 4),
                "called_by": None,
                "reason": "semantic_chunk_match",
                "path": [intent.label],
                "path_order": 0,
                "query_intent_overlap": query_intent_overlap,
                "query_exact_overlap": query_exact_overlap,
                "query_intent_count": len(query_token_embeddings),
                "query_chunk_similarity": round(query_chunk_similarity, 4),
                "intent_query_similarity": round(intent_query_similarity, 4),
                "intent_chunk_similarity": round(intent_chunk_similarity, 4),
                "candidate_label_exact": candidate_label_exact,
                "candidate_intent_matched": candidate_intent_matched,
                "from_seed": False,
                "called_by_exact": False,
            })

        return candidates

    def _llm_intent_chunk_candidates(
        self,
        query_embedding,
        query_intent_set,
        query_token_embeddings,
        query_match_intent_ids,
        top_k_per_intent: int = 4,
    ):
        candidates = []
        seen_pairs = set()

        for label, intent_embedding in query_token_embeddings:
            if not label or not intent_embedding:
                continue

            graph_intent = self.store.find_intent_by_label(label)
            intent = graph_intent or SimpleNamespace(
                intent_id=f"query_intent:{label}",
                label=label,
                type="concept",
                domain="general",
                state="query",
            )

            for label_chunk_similarity, chunk in self.store.search_chunks(intent_embedding, top_k=top_k_per_intent):
                query_chunk_similarity = cosine_similarity(query_embedding, chunk.embedding)
                if label_chunk_similarity < 0.30 and query_chunk_similarity < 0.30:
                    continue

                pair = (label, chunk.chunk_id)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                query_intent_overlap = self._query_intent_overlap(
                    chunk,
                    query_intent_set,
                    query_token_embeddings,
                    query_match_intent_ids,
                )
                query_exact_overlap = self._query_exact_overlap(chunk, query_intent_set)
                candidate_label_exact = label in query_intent_set
                candidate_intent_matched = (
                    (getattr(intent, "intent_id", None) in set(query_match_intent_ids or []))
                    or candidate_label_exact
                )

                score = (
                    label_chunk_similarity * 0.65
                    + query_chunk_similarity * 0.25
                    + min(0.05, chunk.reinforcement_count * 0.01)
                )

                candidates.append({
                    "chunk": chunk,
                    "score": round(score, 4),
                    "layer": 0,
                    "intent": intent,
                    "path_strength": round(label_chunk_similarity, 4),
                    "called_by": None,
                    "reason": "llm_intent_chunk_match",
                    "path": [label],
                    "path_order": 0,
                    "query_intent_overlap": query_intent_overlap,
                    "query_exact_overlap": query_exact_overlap,
                    "query_intent_count": len(query_token_embeddings),
                    "query_chunk_similarity": round(query_chunk_similarity, 4),
                    "intent_query_similarity": 1.0,
                    "intent_chunk_similarity": round(label_chunk_similarity, 4),
                    "candidate_label_exact": candidate_label_exact,
                    "candidate_intent_matched": candidate_intent_matched,
                    "from_seed": True,
                    "called_by_exact": False,
                })

        return candidates

    def _best_chunk_intent(self, query_embedding, chunk):
        best = None
        best_rank = None
        for intent_id in chunk.intent_ids:
            intent = self.store.intents.get(intent_id)
            if not intent or intent.state == "archived":
                continue
            intent_query_similarity = cosine_similarity(query_embedding, intent.embedding)
            intent_chunk_similarity = cosine_similarity(intent.embedding, chunk.embedding)
            rank = (intent_chunk_similarity, intent_query_similarity)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best = (intent, intent_query_similarity, intent_chunk_similarity)
        return best

    def _field_seed_ids(self, layers):
        direct = [
            item["intent"].intent_id
            for item in layers.get(0, [])
            if item.get("reason") == "direct_match"
        ]
        if direct:
            return direct[:8]
        return [item["intent"].intent_id for item in layers.get(0, [])[:5]]

    def _edge_path_score(self, source_intent, target_intent, edge, multiplier: float = 1.0) -> float:
        confidence_factor = 0.75 + min(1.0, edge.confidence) * 0.25
        support_factor = 1.0 + min(0.12, max(0, edge.support_count - 1) * 0.04)
        
        # Universal Jump Brake Multipliers
        role_multipliers = {
            "weak_echo": 0.0,
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

    def _collapse_semantic_duplicates(self, selected_chunks):
        collapsed = []
        for c in selected_chunks:
            is_duplicate = False
            for existing in collapsed:
                if cosine_similarity(c["chunk"].embedding, existing["chunk"].embedding) > 0.85:
                    # Duplicate found, keep the one with higher score
                    if c["score"] > existing["score"]:
                        existing.update(c)
                    is_duplicate = True
                    break
            if not is_duplicate:
                collapsed.append(c)
        return collapsed

    def _select_direct(self, candidates):
        direct = [c for c in candidates if c["layer"] == 0]
        if not direct:
            return []
        best_score = direct[0]["score"]
        effective_best = min(best_score, 1.05)
        filtered = [
            c for c in direct
            if c["score"] >= effective_best * 0.72
            and (
                c.get("query_intent_overlap", 0) > 0 
                or c.get("query_chunk_similarity", 0.0) >= 0.35
                or c.get("reason") == "conversational_priming"
                or (
                    c.get("reason") == "contextual_activation"
                    and c.get("intent_query_similarity", 0.0) >= 0.42
                )
            )
        ]

        unique_l0_intents = len(set(c["intent"].intent_id for c in filtered))
        is_broad_query = unique_l0_intents >= 3

        intent_counts: dict[str, int] = {}
        selected = []
        for c in filtered:
            iid = c["intent"].intent_id
            hub_size = len(self.store.get_chunks_by_intent(iid))
            if hub_size > 5:
                max_per_intent = 2 if is_broad_query else 1
                if c.get("query_chunk_similarity", 0.0) < 0.25:
                    continue
            else:
                max_per_intent = 2
            intent_counts[iid] = intent_counts.get(iid, 0) + 1
            if intent_counts[iid] <= max_per_intent:
                selected.append(c)

        # Semantic Collapse & Token Budgeting (Max 8 direct chunks)
        selected = self._collapse_semantic_duplicates(selected)
        return selected[:8]

    def _select_associated(self, candidates, direct, query_intent_set):
        associated = [c for c in candidates if c["layer"] in [1, 2]]
        if not associated:
            return []

        if direct:
            direct_best = max(c["score"] for c in direct)
            effective_direct_best = min(direct_best, 1.05)
            query_count = max([c.get("query_intent_count", 0) for c in direct + associated] or [0], default=0)
            direct_coverage = max((c.get("query_intent_overlap", 0) for c in direct), default=0) / max(1, query_count)
            selected = [
                c for c in associated
                if (
                    self._associated_is_grounded(c)
                    and (
                        c["score"] >= effective_direct_best * 0.75
                        or c.get("query_intent_overlap", 0) >= 2
                        or (
                            c.get("edge_support", 0) > 1
                            and c.get("intent_chunk_similarity", 0.0) >= 0.30
                        )
                        or (
                            c.get("called_by") in query_intent_set
                            and c.get("edge_confidence", 0.0) >= 0.50
                            and c["score"] >= 0.45
                        )
                    )
                )
                and c["score"] >= 0.40
            ]
            selected = self._collapse_semantic_duplicates(selected)
            l1_selected = [c for c in selected if c["layer"] == 1][:4]
            l2_selected = [c for c in selected if c["layer"] == 2][:2]
            return l1_selected + l2_selected

        best_score = associated[0]["score"]
        effective_best = min(best_score, 1.05)
        selected = [
            c for c in associated
            if c["score"] >= 0.40
            and self._associated_is_grounded(c)
            and (
                (
                    c.get("called_by") in query_intent_set
                    and c.get("edge_confidence", 0.0) >= 0.50
                )
                or (
                    c["score"] >= effective_best * 0.75
                    and (
                        c.get("query_intent_overlap", 0) >= 2
                        or (
                            c.get("query_intent_overlap", 0) >= 1
                            and c.get("edge_confidence", 0.0) >= 0.50
                            and c["score"] >= 0.50
                        )
                        or (
                            c.get("edge_confidence", 0.0) >= 0.50
                            and c["score"] >= 0.55
                        )
                        or (
                            c.get("edge_support", 0) > 1
                            and c.get("intent_chunk_similarity", 0.0) >= 0.30
                        )
                    )
                )
            )
        ]
        selected = self._collapse_semantic_duplicates(selected)
        l1_selected = [c for c in selected if c["layer"] == 1][:4]
        l2_selected = [c for c in selected if c["layer"] == 2][:2]
        return l1_selected + l2_selected

    def _associated_is_grounded(self, item):
        if item.get("query_intent_overlap", 0) >= 2:
            return True
        if item.get("intent_chunk_similarity", 0.0) >= 0.30:
            return True
        if item.get("candidate_intent_matched"):
            return True
        edge_type = item.get("edge_type")
        return edge_type in {"event_link", "causal_link", "instrumental_link"}

    def _dedupe_layer(self, items):
        best = {}
        for item in items:
            intent = item["intent"]
            if intent.intent_id not in best or item["score"] > best[intent.intent_id]["score"]:
                best[intent.intent_id] = item
        return sorted(best.values(), key=lambda x: x["score"], reverse=True)

    def remove_duplicates(self, candidates):
        best_by_chunk = {}
        for item in candidates:
            cid = item["chunk"].chunk_id
            if cid not in best_by_chunk or self._candidate_rank(item) > self._candidate_rank(best_by_chunk[cid]):
                best_by_chunk[cid] = item
        return list(best_by_chunk.values())

    def _candidate_rank(self, item):
        overlap = item.get("query_intent_overlap", 0)
        exact_overlap = item.get("query_exact_overlap", 0)
        strong_direct = 1 if (
            item.get("layer") == 0
            and item.get("reason") == "direct_match"
            and item.get("candidate_label_exact")
            and exact_overlap >= 2
        ) else 0
        direct_grounded = 1 if (
            item.get("layer") == 0
            and (
                item.get("candidate_label_exact")
                or item.get("intent_query_similarity", 0.0) >= 0.42
            )
        ) else 0
        associative = 1 if (
            item.get("called_by")
            and item.get("layer") in {1, 2}
            and exact_overlap < 2
            and item.get("intent_chunk_similarity", 0.0) >= 0.30
        ) else 0
        anchored_assoc = 1 if (
            item.get("called_by_exact")
            and item.get("layer") in {1, 2}
            and (item.get("edge_confidence") or 0.0) >= 0.54
            and item.get("intent_chunk_similarity", 0.0) >= 0.30
            and exact_overlap < 2
        ) else 0
        from_seed = 1 if item.get("from_seed") else 0
        intent_type_priority = 0 if getattr(item.get("intent"), "type", "") == "entity/location" else 1
        centrality = item.get("intent_chunk_similarity", 0.0)
        path_priority = -item.get("path_order", 0)
        return (
            strong_direct,
            anchored_assoc,
            associative,
            direct_grounded,
            from_seed,
            intent_type_priority,
            centrality,
            path_priority,
            item["score"],
        )

    def _matching_intent_ids(self, query_intent_labels, query_token_embeddings=None):
        query_token_embeddings = query_token_embeddings or []
        ids = set()
        for label in query_intent_labels or []:
            intent = self.store.find_intent_by_label(label)
            if intent and intent.state != "archived":
                ids.add(intent.intent_id)
        for _label, token_emb in query_token_embeddings:
            intent = self.store.find_intent_by_embedding(token_emb, threshold=self.intent_match_threshold)
            if intent and intent.state != "archived":
                ids.add(intent.intent_id)
        return ids

    def _query_intent_overlap(self, chunk, query_intent_set, query_token_embeddings=None, query_match_intent_ids=None):
        query_match_intent_ids = set(query_match_intent_ids or [])
        if query_match_intent_ids:
            return len(set(chunk.intent_ids).intersection(query_match_intent_ids))

        query_token_embeddings = query_token_embeddings or []
        if not query_intent_set and not query_token_embeddings:
            return 0
        hits = 0
        for iid in chunk.intent_ids:
            intent = self.store.intents.get(iid)
            if not intent:
                continue
            if intent.label in query_intent_set:
                hits += 1
                continue
            if any(cosine_similarity(intent.embedding, token_emb) >= self.intent_match_threshold for _, token_emb in query_token_embeddings):
                hits += 1
        return hits

    def _query_exact_overlap(self, chunk, query_intent_set):
        if not query_intent_set:
            return 0
        hits = 0
        for iid in chunk.intent_ids:
            intent = self.store.intents.get(iid)
            if intent and intent.label in query_intent_set:
                hits += 1
        return hits
