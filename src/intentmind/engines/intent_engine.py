from __future__ import annotations

import re
import time
import os
import json
from typing import Any, Callable, List, Tuple
from ..store import MemoryStore
from ..models import ExtractedIntent, IntentCandidate
from ..embeddings import BaseEmbedder, cosine_similarity

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# Minimum token length to consider as intent candidate
_MIN_TOKEN_LEN = 3
_GRAPH_ROLES = {"topic", "entity", "fact"}


class IntentEngine:
    def __init__(self, store: MemoryStore, embedder: BaseEmbedder,
                 consolidation_threshold: float = 0.92,
                 intent_similarity_threshold: float = 0.82,
                 enable_llm_enrichment: bool = False,
                 core_extractor: str = "llm",
                 model: str | None = None,
                 extractor: Callable[[str], Any] | None = None,
                 enable_query_llm: bool | None = None):
        self.store = store
        self.embedder = embedder
        self.consolidation_threshold = consolidation_threshold
        self.intent_similarity_threshold = intent_similarity_threshold
        self.enable_llm_enrichment = enable_llm_enrichment
        self.core_extractor = core_extractor
        self.model = model
        self.external_extractor = extractor
        self._label_embedding_cache: dict[str, List[float]] = {}
        self._last_extraction_raw_count = 0
        self._last_llm_edges: list = []  # Cached edges from last LLM extraction
        self._grounded_cache: dict = {}  # { "text": (intents, edges) }
        if enable_query_llm is None:
            enable_query_llm = os.environ.get("INTENTMIND_QUERY_LLM", "1").lower() in {"1", "true", "yes"}
        self.enable_query_llm = enable_query_llm
        
        self.client = None
        wants_llm = self.core_extractor == "llm" or self.enable_llm_enrichment
        if wants_llm and OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


    # ------------------------------------------------------------------ #
    #  EXTRACT: language-agnostic tokenization                            #
    # ------------------------------------------------------------------ #

    def extract_candidates(self, text: str) -> List[IntentCandidate]:
        """
        Deterministic, embedding-based core intent extraction.
        Evaluates candidate tokens and returns scored IntentCandidate objects.
        """
        tokens = re.findall(r"\w+", text, flags=re.UNICODE)
        unique_tokens = []
        seen = set()
        for token in tokens:
            if len(token) >= _MIN_TOKEN_LEN and token.lower() not in seen:
                seen.add(token.lower())
                unique_tokens.append(token)
        
        if not unique_tokens:
            return []

        text_emb = self.embedder.embed(text)
        candidates = []
        
        for token in unique_tokens:
            is_capitalized = token[0].isupper() if token else False
            token_lower = token.lower()
            token_emb = self._embed_label(token_lower)
            semantic_density = cosine_similarity(text_emb, token_emb)
            
            # Simple heuristic for concreteness (longer words often more specific)
            concreteness = min(0.15, len(token) * 0.015)
            if is_capitalized:
                concreteness += 0.25  # Massive bonus for entities/proper nouns
            
            # Graph match bonus
            matched_intent = self.store.find_intent_by_embedding(token_emb, threshold=self.intent_similarity_threshold)
            graph_match = 0.30 if matched_intent else 0.0
            
            # Generic/Noise penalty
            generic_penalty = 0.05 if len(token) <= 3 else 0.0
            
            quality_score = semantic_density + concreteness + graph_match - generic_penalty
            
            # Decision
            if quality_score >= 0.60:
                decision = "reuse" if matched_intent else "create_active"
            elif 0.35 <= quality_score < 0.60:
                decision = "create_candidate"
            else:
                decision = "reject"
                
            candidate = IntentCandidate(
                text=token,
                normalized=token,
                embedding=token_emb,
                intent_type="concept",
                extraction_confidence=1.0,
                quality_score=quality_score,
                concreteness_score=concreteness,
                matched_intent_id=matched_intent.intent_id if matched_intent else None,
                decision=decision,
                reason=f"density={semantic_density:.2f}, match={graph_match:.2f}"
            )
            candidates.append(candidate)
            
        candidates = sorted(candidates, key=lambda x: x.quality_score, reverse=True)
        return candidates

    def extract_intents_with_embeddings(self, text: str) -> List[Tuple[str, List[float]]]:
        """
        Extracts Abstract Cognitive Intents using LLM ONLY IF enable_llm_enrichment=True.
        Otherwise falls back to deterministic extraction.
        Used for INGESTION — standard extraction.
        """
        extracted = self.extract_intents_with_metadata(text)
        if extracted:
            return [(item.text, item.embedding) for item in extracted]
        if self._last_extraction_raw_count > 0:
            return []
            
        # Fallback to returning strings for active deterministic candidates
        result = []
        candidates = self.extract_candidates(text)
        for cand in candidates:
            if cand.decision in ["reuse", "create_active", "create_candidate"]:
                result.append((cand.text, cand.embedding))
        return result

    def extract_query_intents(self, query: str) -> List[Tuple[str, List[float]]]:
        """
        Query-time intent matching.

        Chat recall should not burn multiple model calls before answering.
        The default path is local: match the query against existing graph
        labels by lexical overlap and embeddings. The optional grounded LLM
        path can be enabled with INTENTMIND_QUERY_LLM=1, but it never falls
        through to a second core extraction call.
        """
        # If external extractor is set (e.g. fixture benchmark), use standard path
        if self.external_extractor:
            return self.extract_intents_with_embeddings(query)

        if self.enable_query_llm and self.client and (self.core_extractor == "llm" or self.enable_llm_enrichment):
            grounded = self._extract_grounded_llm(query)
            if grounded:
                result = []
                seen = set()
                for item in grounded:
                    if not self._is_graph_intent(item):
                        continue
                    label = item.label.strip()
                    if not label:
                        continue
                    key = label.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    embedding = self._embed_label(label)
                    result.append((label, embedding))
                if result:
                    return result

        local_matches = self.match_existing_query_intents(query)
        if local_matches:
            return local_matches
        return []

    def match_existing_query_intents(self, query: str, limit: int = 5) -> List[Tuple[str, List[float]]]:
        query_embedding = self.embedder.embed(query)
        query_tokens = self._label_tokens(query)
        matches = []

        for intent in self.store.intents.values():
            if intent.state == "archived":
                continue
            label_tokens = self._label_tokens(intent.label)
            lexical = 0.0
            if label_tokens and query_tokens:
                overlap = len(label_tokens.intersection(query_tokens))
                lexical = overlap / len(label_tokens)
                if label_tokens.issubset(query_tokens):
                    lexical = 1.0

            similarity = cosine_similarity(query_embedding, intent.embedding)
            if lexical >= 0.50 or similarity >= self.intent_similarity_threshold:
                score = max(lexical, similarity)
                matches.append((score, intent))

        matches.sort(key=lambda item: item[0], reverse=True)
        return [(intent.label, intent.embedding) for _score, intent in matches[:limit]]

    def extract_intents_with_metadata(self, text: str) -> List[IntentCandidate]:
        # Fast path: Did we just extract this exact text during query?
        if self._grounded_cache and self._grounded_cache.get("text") == text.strip():
            extracted = self._grounded_cache["intents"]
            self._last_llm_edges = self._grounded_cache["edges"]
        else:
            extracted = self._extract_with_external(text)
            if not extracted and (self.core_extractor == "llm" or self.enable_llm_enrichment):
                extracted = self._extract_with_llm(text)

        self._last_extraction_raw_count = len(extracted)

        result = []
        seen = set()
        for item in extracted:
            label = item.label.strip()
            if not label or not self._is_graph_intent(item):
                continue
            key = label.casefold()
            if key in seen:
                continue
            seen.add(key)
            embedding = self._embed_label(label)
            confidence = self._clamp_confidence(item.confidence)
            result.append(
                IntentCandidate(
                    text=label,
                    normalized=label,
                    embedding=embedding,
                    intent_type=self._normalize_intent_type(item.type),
                    domain=item.domain,
                    role=item.role,
                    extraction_confidence=confidence,
                    quality_score=confidence,
                    decision="create_active" if confidence >= 0.55 else "create_candidate",
                    reason=item.reason or item.source,
                )
            )
            if len(result) >= 8:
                break
        return result

    def _extract_with_external(self, text: str) -> List[ExtractedIntent]:
        if not self.external_extractor:
            return []
        try:
            raw = self.external_extractor(text)
        except Exception:
            return []
        return self._coerce_extracted_intents(raw, source="external")

    def _extract_with_llm(self, text: str) -> List[ExtractedIntent]:
        if not self.client or not self.model:
            return []
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an intent extraction engine for a COGNITIVE GRAPH.\n\n"
                            "Task:\n"
                            "Decompose the message into ATOMIC concept nodes AND their relationships.\n\n"
                            "CRITICAL RULES FOR INTENTS:\n"
                            "- Each intent MUST be a SINGLE WORD (1 word). This is NON-NEGOTIABLE.\n"
                            "- The ONLY exception: proper nouns with 2 words (e.g., 'İstanbul', 'Merkez Bankası').\n"
                            "- WRONG: 'arabamla gitmek' → RIGHT: 'araba' and 'gitmek' as SEPARATE intents.\n"
                            "- WRONG: 'benzin fiyatı' → RIGHT: 'benzin' and 'fiyat' as SEPARATE intents.\n"
                            "- Extract VERBS as bare infinitive/root form (e.g., 'gidiyorum' → 'gitmek').\n"
                            "- Extract NOUNS as bare singular form (e.g., 'arabayla' → 'araba').\n"
                            "- Intents MUST be in the SAME LANGUAGE as the input.\n"
                            "- ALWAYS extract proper nouns as entity intents.\n"
                            "- Ignore conversational filler (selam, nasılsın, evet, hayır, tamam).\n"
                            "- Maximum 6 intents per message.\n"
                            "- Output JSON only. No explanation.\n\n"
                            "RULES FOR EDGES:\n"
                            "- Define the relationship between each pair of related intents.\n"
                            "- Edge types: instrumental (tool/vehicle for action), spatial (location/destination), "
                            "causal (cause-effect), temporal (time relation), thematic (same topic), "
                            "possessive (ownership), descriptive (attribute/property).\n"
                            "- Only create edges between intents that have a REAL relationship in the sentence.\n"
                            "- Do NOT create edges between unrelated intents.\n\n"
                            "Output format:\n"
                            "{\n"
                            "  \"intents\": [\n"
                            "    {\"label\": \"word\", \"type\": \"concept|entity|action|fact\", \"role\": \"topic|entity|fact\", \"domain\": \"Domain\"}\n"
                            "  ],\n"
                            "  \"edges\": [\n"
                            "    {\"from\": \"word_a\", \"to\": \"word_b\", \"type\": \"instrumental|spatial|causal|temporal|thematic|possessive|descriptive\"}\n"
                            "  ]\n"
                            "}"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            # Cache LLM-extracted edges for use in ingest()
            self._last_llm_edges = data.get("edges", [])
            return self._coerce_extracted_intents(data, source="llm")
        except Exception:
            self._last_llm_edges = []
            return []

    def _extract_grounded_llm(self, query: str) -> List[ExtractedIntent]:
        """
        Query-time grounded extraction: give the LLM the list of existing
        intent labels from the graph and ask which are relevant to this query.
        Also asks for any NEW concepts not in the existing list.

        This eliminates the wording mismatch problem where the LLM uses
        different phrases at ingest time vs query time.
        """
        if not self.client or not self.model:
            return []

        # Collect existing intent labels (only active/weak, skip archived)
        # Sort by energy descending so the most relevant intents are always
        # included even when we hit the token-budget cap.
        existing_intents = [
            intent
            for intent in self.store.intents.values()
            if intent.state in ("active", "weak")
        ]
        if not existing_intents:
            # No graph yet, fall back to standard extraction
            return self._extract_with_llm(query)

        existing_intents.sort(key=lambda i: i.energy, reverse=True)
        labels_str = ", ".join(i.label for i in existing_intents[:100])

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a memory recall assistant for a COGNITIVE GRAPH. The user has a query and a knowledge graph "
                            "with known concept labels.\n\n"
                            "Your job:\n"
                            "1. Identify which EXISTING concepts are directly referenced, mentioned, "
                            "or closely implied by the query.\n"
                            "2. List any genuinely NEW concepts in the query not covered by existing labels.\n\n"
                            "CRITICAL RULES FOR INTENTS:\n"
                            "- Each intent MUST be a SINGLE WORD (1 word). This is NON-NEGOTIABLE.\n"
                            "- The ONLY exception: proper nouns with 2 words (e.g., 'İstanbul', 'Merkez Bankası').\n"
                            "- WRONG: 'arabamla gitmek' → RIGHT: 'araba' and 'gitmek' as SEPARATE intents.\n"
                            "- WRONG: 'benzin fiyatı' → RIGHT: 'benzin' and 'fiyat' as SEPARATE intents.\n"
                            "- Extract VERBS as bare infinitive/root form (e.g., 'gidiyorum' → 'gitmek', 'aldım' → 'almak').\n"
                            "- Extract NOUNS as bare singular form (e.g., 'arabayla' → 'araba', 'benzine' → 'benzin').\n"
                            "- Intents MUST be in the EXACT SAME LANGUAGE as the user's input.\n"
                            "- ALWAYS extract proper nouns, brand names, and specific places as exact entity intents.\n"
                            "- Ignore conversational filler.\n"
                            "- Preserve contextual entity meaning. Do not reinterpret entities outside conversation context.\n"
                            "- Only match concepts the query DIRECTLY talks about or strongly implies as durable subjects.\n"
                            "- Do not match request operators as graph concepts.\n"
                            "- Assign role='topic' for durable subjects, role='entity' for named things/places/brands, role='fact' for concrete facts.\n"
                            "- Assign role='task' or role='modifier' for request shape (these won't become graph nodes).\n"
                            "- For 'matched', return labels EXACTLY as they appear in the existing list.\n"
                            "- CRITICAL: If the query contains '--- Conversation History ---', extract intents ONLY for the '--- Current Query ---'.\n"
                            "- Assign a thematic 'domain' for each intent (e.g., 'Finance', 'Oncology', 'Military').\n"
                            "- Return at most 5 matched + 3 new.\n\n"
                            "RULES FOR EDGES:\n"
                            "- Define the relationship between each pair of related intents.\n"
                            "- Edge types: instrumental, spatial, causal, temporal, thematic, possessive, descriptive.\n"
                            "- Use ONLY the labels you extracted in 'matched' or 'new'.\n\n"
                            "Return JSON:\n"
                            "{\"matched\": [{\"label\": \"single_word\", \"confidence\": 0.0, \"role\": \"topic|entity|fact\", \"domain\": \"...\"}], "
                            "\"new\": [{\"label\": \"single_word\", \"type\": \"concept|entity|action|fact\", \"role\": \"topic|entity|fact\", \"confidence\": 0.0, \"domain\": \"...\"}], "
                            "\"edges\": [{\"from\": \"word_a\", \"to\": \"word_b\", \"type\": \"instrumental|spatial|causal|temporal|thematic|possessive|descriptive\"}], "
                            "\"request\": {\"role\": \"task|modifier\", \"description\": \"...\"}}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Existing concepts: [{labels_str}]\n\n"
                            f"Query: {query}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            content = response.choices[0].message.content
            data = json.loads(content)

            results: List[ExtractedIntent] = []

            # Process matched existing intents (confidence >= 0.80)
            matched_count = 0
            for item in data.get("matched", []):
                label = item.get("label", "") if isinstance(item, dict) else str(item)
                confidence = item.get("confidence", 0.90) if isinstance(item, dict) else 0.90
                domain = item.get("domain", "general") if isinstance(item, dict) else "general"
                role = item.get("role", "topic") if isinstance(item, dict) else "topic"
                confidence = self._clamp_confidence(confidence)
                extracted = ExtractedIntent(
                        label=label.strip(),
                        type="concept",
                        domain=domain,
                        role=self._normalize_semantic_role(role),
                        confidence=confidence,
                        source="llm_grounded",
                        reason="matched existing intent",
                    )
                if extracted.label and self._is_graph_intent(extracted) and confidence >= 0.80 and matched_count < 5:
                    results.append(extracted)
                    matched_count += 1

            # Process new intents (confidence >= 0.75)
            new_count = 0
            for item in data.get("new", []):
                if isinstance(item, dict):
                    label = item.get("label", "")
                    confidence = item.get("confidence", 0.80)
                    intent_type = item.get("type", "concept")
                    role = item.get("role", "topic")
                    domain = item.get("domain", "general")
                elif isinstance(item, str):
                    label = item
                    confidence = 0.80
                    intent_type = "concept"
                    role = "topic"
                    domain = "general"
                else:
                    continue
                confidence = self._clamp_confidence(confidence)
                extracted = ExtractedIntent(
                        label=label.strip(),
                        type=intent_type,
                        domain=domain,
                        role=self._normalize_semantic_role(role),
                        confidence=confidence,
                        source="llm_grounded_new",
                        reason="new concept from query",
                    )
                if extracted.label and self._is_graph_intent(extracted) and confidence >= 0.70 and new_count < 3:
                    results.append(extracted)
                    new_count += 1

            # Cache the results for ingestion so we don't call LLM again!
            # The query string format from api.py has the raw text at the end:
            raw_text = query.split("--- Current Query (EXTRACT INTENTS PRIMARILY FOR THIS) ---\nuser: ")[-1] if "--- Current Query" in query else query
            self._grounded_cache = {
                "text": raw_text.strip(),
                "intents": results,
                "edges": data.get("edges", [])
            }

            return results
        except Exception:
            # Fallback to standard extraction on any error
            return self._extract_with_llm(query)

    def _coerce_intent_labels(self, raw: Any) -> List[str]:
        return [item.label for item in self._coerce_extracted_intents(raw)]

    def _label_tokens(self, label: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"\w+", label or "", flags=re.UNICODE)
            if len(token) >= _MIN_TOKEN_LEN
        }

    def _is_graph_intent(self, item: ExtractedIntent | IntentCandidate) -> bool:
        return self._normalize_semantic_role(getattr(item, "role", "topic")) in _GRAPH_ROLES

    def _normalize_semantic_role(self, role: Any) -> str:
        value = str(role or "topic").strip().lower().replace("_", "-")
        aliases = {
            "concept": "topic",
            "subject": "topic",
            "node": "topic",
            "place": "entity",
            "location": "entity",
            "request": "task",
            "instruction": "task",
            "operator": "modifier",
            "style": "modifier",
        }
        value = aliases.get(value, value)
        allowed = {"topic", "entity", "fact", "task", "modifier"}
        return value if value in allowed else "topic"

    def _coerce_extracted_intents(self, raw: Any, source: str = "extractor") -> List[ExtractedIntent]:
        if raw is None:
            return []
        if isinstance(raw, dict):
            raw = raw.get("intents", [])
        if not isinstance(raw, list):
            return []

        items = []
        seen = set()
        for raw_item in raw:
            item_type = "concept"
            domain = "general"
            role = "topic"
            confidence = 1.0
            reason = ""
            if isinstance(raw_item, str):
                label = raw_item
            elif isinstance(raw_item, dict):
                label = raw_item.get("label") or raw_item.get("text") or raw_item.get("name")
                item_type = raw_item.get("type") or raw_item.get("intent_type") or item_type
                domain = raw_item.get("domain", domain)
                role = raw_item.get("role") or raw_item.get("semantic_role") or raw_item.get("intent_role")
                if role is None and str(item_type).strip().lower() in {"task", "modifier", "operator", "request"}:
                    role = item_type
                role = role or "topic"
                confidence = raw_item.get("confidence", confidence)
                reason = raw_item.get("reason", "")
            else:
                continue

            label = str(label).strip()
            if not label:
                continue
            key = label.casefold()
            if key in seen:
                continue
            seen.add(key)
            items.append(
                ExtractedIntent(
                    label=label,
                    type=self._normalize_intent_type(item_type),
                    domain=str(domain).strip(),
                    role=self._normalize_semantic_role(role),
                    confidence=self._clamp_confidence(confidence),
                    source=source,
                    reason=str(reason).strip(),
                )
            )
            if len(items) >= 8:
                break
        return items

    def _normalize_intent_type(self, intent_type: Any) -> str:
        value = str(intent_type or "concept").strip().lower().replace("_", "/")
        allowed = {
            "concept",
            "entity",
            "entity/location",
            "location",
            "action",
            "topic",
            "fact",
            "abstract",
        }
        if value == "location":
            return "entity/location"
        return value if value in allowed else "concept"

    def _clamp_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 1.0
        return max(0.0, min(1.0, confidence))

    def _embed_label(self, label: str) -> List[float]:
        key = label.strip().casefold()
        if key not in self._label_embedding_cache:
            self._label_embedding_cache[key] = self.embedder.embed(label)
        return self._label_embedding_cache[key]

    def _find_reusable_intent(self, cand: IntentCandidate):
        exact = self.store.find_intent_by_label(cand.text)
        if exact and exact.state != "archived":
            return exact

        semantic = self.store.find_intent_by_embedding(cand.embedding, threshold=self.intent_similarity_threshold)
        if not semantic or semantic.state == "archived":
            return None

        sim = cosine_similarity(cand.embedding, semantic.embedding)
        if sim >= max(0.90, self.intent_similarity_threshold + 0.05):
            return semantic

        cand_domain = (cand.domain or "general").strip().casefold()
        existing_domain = (semantic.domain or "general").strip().casefold()
        if cand_domain != "general" and cand_domain == existing_domain:
            return semantic

        if self._label_token_overlap(cand.text, semantic.label) > 0:
            return semantic

        return None

    def _label_token_overlap(self, left: str, right: str) -> int:
        left_tokens = {
            token.casefold()
            for token in re.findall(r"\w+", left, flags=re.UNICODE)
            if len(token) >= _MIN_TOKEN_LEN
        }
        right_tokens = {
            token.casefold()
            for token in re.findall(r"\w+", right, flags=re.UNICODE)
            if len(token) >= _MIN_TOKEN_LEN
        }
        return len(left_tokens.intersection(right_tokens))

    def extract_intents(self, text: str) -> List[str]:
        """
        Extract candidate intent tokens from text.
        Delegates to extract_intents_with_embeddings for semantic filtering.
        """
        items = self.extract_intents_with_embeddings(text)
        return [token for token, emb in items]

    # ------------------------------------------------------------------ #
    #  INGEST: embedding-based intent matching                            #
    # ------------------------------------------------------------------ #

    def ingest(self, text: str, source: str = "chat", chunk_id: str | None = None):
        """
        Ingest text into the memory store using deterministic candidate evaluation.
        """
        intent_ids: List[str] = []
        now = time.time()
        
        if self.core_extractor == "llm":
            # 1. LLM Core Extraction
            llm_intents = self.extract_intents_with_metadata(text)
            for cand in llm_intents:
                existing = self._find_reusable_intent(cand)
                if existing:
                    self._reinforce_intent(existing, cand, now, boost=0.10)
                    intent_ids.append(existing.intent_id)
                else:
                    intent = self.store.add_intent(
                        label=cand.text,
                        intent_type=cand.intent_type,
                        embedding=cand.embedding,
                        extraction_confidence=cand.extraction_confidence,
                    )
                    intent.domain = cand.domain
                    intent.source_count = 1
                    intent.state = "active" if cand.extraction_confidence >= 0.55 else "candidate"
                    intent_ids.append(intent.intent_id)
            
            # Optional fallback only when extraction produced nothing at all.
            # If the extractor returned only non-graph/request roles, do not
            # invent token nodes from the raw text.
            if not intent_ids and self._last_extraction_raw_count == 0:
                candidates = self.extract_candidates(text)
                for cand in candidates:
                    if cand.decision == "reuse":
                        intent = self._find_reusable_intent(cand)
                        if intent:
                            self._reinforce_intent(intent, cand, now, boost=0.05)
                            intent_ids.append(intent.intent_id)
                    elif cand.decision in ["create_active", "create_candidate"]:
                        intent = self.store.add_intent(
                            label=cand.text,
                            intent_type=cand.intent_type,
                            embedding=cand.embedding,
                            extraction_confidence=cand.extraction_confidence,
                        )
                        intent.domain = cand.domain
                        intent.source_count = 1
                        intent.state = "active" if cand.decision == "create_active" else "candidate"
                        intent_ids.append(intent.intent_id)
        else:
            # 2. Deterministic Core Extraction
            candidates = self.extract_candidates(text)
            for cand in candidates:
                if cand.decision == "reject":
                    continue
                    
                if cand.decision == "reuse":
                    intent = self._find_reusable_intent(cand)
                    if intent:
                        self._reinforce_intent(intent, cand, now, boost=0.05)
                        intent_ids.append(intent.intent_id)
                elif cand.decision in ["create_active", "create_candidate"]:
                    intent = self.store.add_intent(
                        label=cand.text,
                        intent_type=cand.intent_type,
                        embedding=cand.embedding,
                        extraction_confidence=cand.extraction_confidence,
                    )
                    intent.domain = cand.domain
                    intent.source_count = 1
                    intent.state = "active" if cand.decision == "create_active" else "candidate"
                    intent_ids.append(intent.intent_id)
                    
            # Optional LLM Enrichment on top of deterministic
            if self.enable_llm_enrichment:
                llm_intents = self.extract_intents_with_metadata(text)
                for cand in llm_intents:
                    existing = self._find_reusable_intent(cand)
                    if existing:
                        self._reinforce_intent(existing, cand, now, boost=0.03)
                        intent_ids.append(existing.intent_id)
                    else:
                        intent = self.store.add_intent(
                            label=cand.text,
                            intent_type=cand.intent_type if cand.intent_type != "concept" else "abstract",
                            embedding=cand.embedding,
                            extraction_confidence=cand.extraction_confidence,
                        )
                        intent.domain = cand.domain
                        intent.source_count = 1
                        intent.state = "weak"
                        intent_ids.append(intent.intent_id)

        # Deduplicate intent_ids (same intent matched by different tokens)
        intent_ids = list(dict.fromkeys(intent_ids))

        embedding = self.embedder.embed(text)
        existing_chunk = self.find_similar_chunk(embedding, intent_ids)
        if existing_chunk:
            existing_chunk.reinforcement_count += 1
            existing_chunk.feedback_score = min(1.0, existing_chunk.feedback_score + 0.05)
            for iid in intent_ids:
                if iid not in existing_chunk.intent_ids:
                    existing_chunk.intent_ids.append(iid)
            self._create_edges_from_llm_or_heuristic(intent_ids, evidence_chunk_id=existing_chunk.chunk_id)
            return existing_chunk

        chunk = self.store.add_chunk(
            text=text, summary=text[:240], embedding=embedding,
            intent_ids=intent_ids, source=source, chunk_id=chunk_id,
        )
        self._create_edges_from_llm_or_heuristic(intent_ids, evidence_chunk_id=chunk.chunk_id)
        return chunk

    def _reinforce_intent(self, intent, candidate: IntentCandidate, now: float, boost: float):
        intent.source_count += 1
        scaled_boost = boost * max(0.35, candidate.extraction_confidence)
        intent.energy = min(1.0, intent.energy + scaled_boost)
        intent.extraction_confidence = max(intent.extraction_confidence, candidate.extraction_confidence)
        if intent.domain == "general" and candidate.domain and candidate.domain != "general":
            intent.domain = candidate.domain
        if intent.type == "concept" and candidate.intent_type not in {"concept", ""}:
            intent.type = candidate.intent_type
        intent.last_active = now

    # ------------------------------------------------------------------ #
    #  INTENT TYPE (simplified)                                           #
    # ------------------------------------------------------------------ #

    def intent_type(self, label: str) -> str:
        return "concept"

    # ------------------------------------------------------------------ #
    #  CHUNK SIMILARITY                                                   #
    # ------------------------------------------------------------------ #

    def find_similar_chunk(self, embedding: List[float], intent_ids: List[str]):
        best = None
        best_score = -1.0
        intent_set = set(intent_ids)
        for chunk in self.store.chunks.values():
            overlap_ratio = self._intent_overlap_ratio(intent_set, set(chunk.intent_ids))
            if overlap_ratio < 0.60:
                continue
            sim = cosine_similarity(embedding, chunk.embedding)
            if sim > best_score:
                best = chunk
                best_score = sim
        return best if best and best_score >= self.consolidation_threshold else None

    def _intent_overlap_ratio(self, a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a.intersection(b)) / len(a.union(b))

    # ------------------------------------------------------------------ #
    #  EDGE CREATION                                                      #
    # ------------------------------------------------------------------ #

    def _classify_edge_heuristic(self, a, b, text: str) -> Tuple[str, float]:
        a_type = (a.type or "concept").lower()
        b_type = (b.type or "concept").lower()
        
        if a_type == "entity/location" and b_type == "entity/location":
            return "spatial_link", 0.80
            
        if a_type == "action" and b_type == "action":
            if text and any(w in text.lower() for w in ["because", "due to", "caused", "resulted", "nedeniyle", "sebep"]):
                return "causal_link", 0.85
            return "temporal_link", 0.65
            
        if a_type == "entity" and b_type == "entity":
            return "actor_link", 0.70
            
        if (a_type == "action" and b_type == "entity/location") or (b_type == "action" and a_type == "entity/location"):
            return "event_link", 0.70
            
        if hasattr(a, "domain") and hasattr(b, "domain") and a.domain != "general" and a.domain == b.domain:
            return "thematic_link", 0.65
            
        return "co_occurrence_link", 0.62

    def _create_edges_from_llm_or_heuristic(self, intent_ids: List[str], evidence_chunk_id: str | None = None):
        """Use LLM-extracted typed edges when available, fall back to heuristic."""
        llm_edges = self._last_llm_edges
        self._last_llm_edges = []  # Consume once

        if not llm_edges:
            # Fallback to heuristic
            self.create_edges(intent_ids, evidence_chunk_id=evidence_chunk_id)
            return

        # Build label→intent_id lookup
        label_to_id = {}
        for iid in intent_ids:
            intent = self.store.intents.get(iid)
            if intent:
                label_to_id[intent.label.lower()] = iid

        # LLM edge type → store edge type mapping
        type_map = {
            "instrumental": "instrumental_link",
            "spatial": "spatial_link",
            "causal": "causal_link",
            "temporal": "temporal_link",
            "thematic": "thematic_link",
            "possessive": "actor_link",
            "descriptive": "thematic_link",
        }

        created = set()
        for edge_def in llm_edges:
            from_label = edge_def.get("from", "").lower()
            to_label = edge_def.get("to", "").lower()
            edge_type_raw = edge_def.get("type", "thematic")

            from_id = label_to_id.get(from_label)
            to_id = label_to_id.get(to_label)

            if not from_id or not to_id or from_id == to_id:
                continue

            edge_key = tuple(sorted([from_id, to_id]))
            if edge_key in created:
                continue
            created.add(edge_key)

            edge_type = type_map.get(edge_type_raw, "thematic_link")

            # Weight based on edge type quality
            type_weights = {
                "instrumental_link": 0.80,
                "spatial_link": 0.75,
                "causal_link": 0.85,
                "temporal_link": 0.65,
                "thematic_link": 0.70,
                "actor_link": 0.70,
            }
            weight = type_weights.get(edge_type, 0.65)

            self.store.add_edge(
                from_id, to_id,
                edge_type=edge_type,
                weight=weight,
                evidence_chunk_id=evidence_chunk_id,
            )

    def create_edges(self, intent_ids: List[str], evidence_chunk_id: str | None = None):
        chunk_text = ""
        if evidence_chunk_id:
            chunk = self.store.chunks.get(evidence_chunk_id)
            if chunk:
                chunk_text = chunk.text

        for i in range(len(intent_ids)):
            for j in range(i + 1, len(intent_ids)):
                a = self.store.intents[intent_ids[i]]
                b = self.store.intents[intent_ids[j]]
                sim = cosine_similarity(a.embedding, b.embedding)

                # Gate: Don't create edges between semantically unrelated intents.
                # This is the #1 source of recall noise — batch co-occurrence
                # creates false connections (e.g. "benzin" ↔ "Bursaspor").
                if sim < 0.35:
                    continue

                hub_overlap = max(a.hub_score, b.hub_score)
                noise_risk = max(a.noise_score, b.noise_score)

                # Rebalanced formula: similarity is now dominant (0.45 weight)
                edge_score = (
                    sim * 0.45
                    + 0.25              # baseline co-presence
                    + 0.10              # structural bonus
                    - hub_overlap * 0.15
                    - noise_risk * 0.10
                )
                
                if edge_score >= 0.55:
                    edge_type, type_weight = self._classify_edge_heuristic(a, b, chunk_text)

                    # Extra gate for co_occurrence: require higher similarity
                    if edge_type == "co_occurrence_link" and sim < 0.40:
                        continue

                    weight = min(edge_score, type_weight)
                    
                    self.store.add_edge(
                        a.intent_id,
                        b.intent_id,
                        edge_type=edge_type,
                        weight=round(weight, 3),
                        evidence_chunk_id=evidence_chunk_id,
                    )

    # ------------------------------------------------------------------ #
    #  CONSOLIDATION                                                      #
    # ------------------------------------------------------------------ #

    def consolidate_memory(self) -> dict:
        """Background consolidation to merge redundant chunks and prune dead edges/intents."""
        merged_chunks = 0
        pruned_edges = 0
        pruned_intents = 0

        # Prune archived edges
        dead_edges = [eid for eid, edge in self.store.edges.items()
                      if edge.state == "archived" or edge.energy < 0.10]
        for eid in dead_edges:
            self.store.edges.pop(eid, None)
            pruned_edges += 1
        if pruned_edges:
            self.store._invalidate_neighbor_cache()

        # Prune archived intents that have no chunks
        dead_intents = []
        for iid, intent in self.store.intents.items():
            if intent.state == "archived":
                chunks = self.store.get_chunks_by_intent(iid)
                if not chunks:
                    dead_intents.append(iid)

        for iid in dead_intents:
            self.store.intents.pop(iid, None)
            to_remove_edges = [e for e in self.store.edges.values()
                               if e.source_id == iid or e.target_id == iid]
            for e in to_remove_edges:
                self.store.edges.pop(e.edge_id, None)
            pruned_intents += 1
        if pruned_intents:
            self.store._invalidate_neighbor_cache()

        # Merge highly similar chunks
        chunk_list = list(self.store.chunks.values())
        to_delete = set()
        for i in range(len(chunk_list)):
            c1 = chunk_list[i]
            if c1.chunk_id in to_delete:
                continue
            for j in range(i + 1, len(chunk_list)):
                c2 = chunk_list[j]
                if c2.chunk_id in to_delete:
                    continue
                if self._intent_overlap_ratio(set(c1.intent_ids), set(c2.intent_ids)) >= 0.60:
                    sim = cosine_similarity(c1.embedding, c2.embedding)
                    if sim > 0.95:
                        c1.reinforcement_count += c2.reinforcement_count
                        c1.feedback_score = min(1.0, c1.feedback_score + c2.feedback_score)
                        for iid in c2.intent_ids:
                            if iid not in c1.intent_ids:
                                c1.intent_ids.append(iid)
                        to_delete.add(c2.chunk_id)
                        merged_chunks += 1

        for cid in to_delete:
            self.store.chunks.pop(cid, None)

        return {
            "merged_chunks": merged_chunks,
            "pruned_edges": pruned_edges,
            "pruned_intents": pruned_intents,
        }
