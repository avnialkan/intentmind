import os
import json
import time
import logging
from typing import List, Dict, Any
from ..store import MemoryStore
from ..models import Chunk, IntentNode
from ..embeddings import BaseEmbedder

logger = logging.getLogger(__name__)

class ConsolidationEngine:
    """
    Analyzes episodic memory chunks to synthesize long-term semantic facts.
    Handles contradictions by archiving outdated memories.
    """
    def __init__(self, store: MemoryStore, embedder: BaseEmbedder, client=None, model: str = None):
        self.store = store
        self.embedder = embedder
        self.client = client
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def consolidate(self, batch_size: int = 10) -> int:
        """
        Runs the consolidation loop on recent un-consolidated episodic chunks.
        Returns the number of semantic facts generated.
        """
        if not self.client:
            logger.warning("Consolidation requires OpenAI client.")
            return 0

        # Find recent episodic chunks that haven't been archived or made semantic
        # In a real system, we might track 'last_consolidated' timestamp.
        # Here we just grab the top 20 most recent episodic chunks.
        episodic_chunks = [
            c for c in self.store.chunks.values() 
            if c.memory_tier == "episodic" and getattr(c, "source", "") == "chat"
        ]
        episodic_chunks = sorted(episodic_chunks, key=lambda x: x.created_at, reverse=True)[:batch_size]

        if len(episodic_chunks) < 3:
            return 0 # Not enough context to synthesize patterns

        # Group by intents (topologically) to find related memories
        # For simplicity, we pass them all to the LLM if batch is small.
        chunk_texts = []
        for i, c in enumerate(episodic_chunks):
            chunk_texts.append(f"[{i}] (ID: {c.chunk_id}, Intent IDs: {','.join(c.intent_ids)}) {c.text}")

        prompt = (
            "You are a Cognitive Consolidation Engine for an AI memory system.\n"
            "Your job is to analyze recent episodic memories (raw interactions) and synthesize them into "
            "permanent, generalized 'semantic' facts about the user or the world.\n\n"
            "RULES:\n"
            "1. Consolidate specific events into general preferences or facts (e.g. 'I bought a BMW yesterday' + 'My car is blue' -> 'User owns a blue BMW').\n"
            "2. Identify any CONTRADICTIONS (e.g. older memory: 'I hate tea', newer memory: 'I love tea'). The newer memory always wins. Mark the older chunk to be archived.\n"
            "3. Output ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "semantic_facts": [\n'
            '    { "fact": "synthesized fact string", "related_chunk_ids": ["chunk_id_1", "chunk_id_2"] }\n'
            "  ],\n"
            '  "contradictions": [\n'
            '    { "archived_chunk_id": "id of the old incorrect chunk", "reason": "brief explanation" }\n'
            "  ]\n"
            "}\n\n"
            "EPISODIC MEMORIES:\n" + "\n".join(chunk_texts)
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            raw_json = response.choices[0].message.content
            data = json.loads(raw_json)
        except Exception as e:
            logger.error(f"Consolidation LLM failed: {e}")
            return 0

        facts_created = 0

        # Process Contradictions
        for contra in data.get("contradictions", []):
            old_id = contra.get("archived_chunk_id")
            if old_id and old_id in self.store.chunks:
                self.store.chunks[old_id].memory_tier = "archived"
                self.store.chunks[old_id].feedback_score -= 1.0 # Penalize trust
                logger.info(f"Contradiction handled: Archived {old_id}. Reason: {contra.get('reason')}")

        # Process New Semantic Facts
        for fact_item in data.get("semantic_facts", []):
            fact_text = fact_item.get("fact")
            related_ids = fact_item.get("related_chunk_ids", [])
            if not fact_text:
                continue
                
            # Create a new Semantic Chunk
            emb = self.embedder.embed(fact_text)
            
            # Aggregate intent IDs from related chunks
            intent_ids = set()
            for rid in related_ids:
                if rid in self.store.chunks:
                    intent_ids.update(self.store.chunks[rid].intent_ids)
                    # Upgrade the episodic chunks to "archived" since they are now consolidated
                    self.store.chunks[rid].memory_tier = "archived"
            
            chunk = self.store.add_chunk(
                text=fact_text,
                summary=fact_text,
                embedding=emb,
                intent_ids=list(intent_ids),
                source="consolidation"
            )
            chunk.memory_tier = "semantic"
            chunk.reinforcement_count = max(2, len(related_ids)) # Inherit trust
            facts_created += 1
            
        return facts_created
