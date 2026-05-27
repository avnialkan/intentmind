from __future__ import annotations

from .answer_focus_builder import AnswerFocusBuilder

class PromptBuilder:
    """
    THE most critical component of Intentmind.
    
    Everything — intent extraction, edge creation, recall engine,
    energy propagation — exists to produce THIS single output.
    
    Available data per recalled memory item:
    ─────────────────────────────────────────
    Intent (IntentNode):
      .label           → "akıllı robot süpürge"
      .type            → "concept" | "entity" | "action" | "entity/location"
      .domain          → "Technology" | "Finance" | "general"
      .energy          → 0.0–1.0 (activation strength)
      .state           → "active" | "candidate" | "weak"
      .source_count    → how many times this intent was seen
    
    Edge (if associated/echo):
      edge_type        → "thematic_link" | "causal_link" | "co_occurrence_link" | ...
      edge_weight      → 0.0–1.0
      edge_confidence  → 0.0–1.0
      edge_support     → int (how many chunks reinforce this edge)
      called_by        → parent intent label that activated this one
      path             → ["akıllı robot süpürge", "internet güvenliği"]
    
    Chunk (memory content):
      .text            → actual content
      .source          → "news" | "chat" | "user" | "ai"
      .intent_ids      → which intents this chunk belongs to
    
    Layer info:
      layer            → 0 (direct), 1 (associated), 2 (weak_echo), 3 (faint_echo)
      score            → relevance score
      reason           → "direct_match" | "neighbor_intent_reactivation" | ...
    
    Cognitive Field:
      seed_intents     → intents directly activated by query
      activated_intents → intents that resonated through graph
    """
    
    def __init__(self, max_chars: int = 60000):
        self.max_chars = max_chars
        self.answer_focus = AnswerFocusBuilder()

    # ================================================================== #
    #  PUBLIC API                                                         #
    # ================================================================== #

    def build(self, user_query, recall_result, cognitive_state):
        role = self._build_role()
        
        request_desc = recall_result.get("request_description")
        if request_desc:
            request_block = f"[USER INTENT SUMMARY]\n{request_desc}\n\n"
        else:
            request_block = ""
            
        graph = self._build_graph_state(recall_result)
        focus = self.answer_focus.build_block(user_query, recall_result)
        memories = self._build_memories(recall_result)
        
        prompt = f"{role}\n\n{request_block}{graph}\n\n{focus}\n\n{memories}".strip()
        
        if len(prompt) <= self.max_chars:
            return prompt
        return prompt[:self.max_chars] + "\n[TRIMMED]"

    # ================================================================== #
    #  [1] ROLE — 3 lines, zero waste                                     #
    # ================================================================== #

    def _build_role(self):
        return (
            "You are an intelligent assistant with a long-term associative memory.\n"
            "The user has previously shared information with you, which has been recalled via a Cognitive Graph.\n\n"
            "RESPONSE PROCEDURE:\n"
            "1. Identify what the user is asking for now: answer, guess, follow-up continuation, advice, or system/language explanation.\n"
            "2. Read [ANSWER FOCUS] first. If it contains a primary_candidate relevant to the question, make that the lead of your answer.\n"
            "3. If [ANSWER FOCUS] contains current_observation, treat it as the live user situation and answer it directly even when no long-term memory was recalled.\n"
            "4. Use [RECALLED FACTS] as long-term evidence. Direct facts are strongest; associated facts can be used when they clearly support the current question; weak echoes are hints only.\n"
            "5. Use [COGNITIVE FIELD] only for navigation. It is NOT evidence and must never be treated as a fact.\n"
            "6. Then answer in the user's language, briefly and conversationally.\n\n"
            "HARD BEHAVIOR RULES:\n"
            "- Answer the user's current message directly; do not turn ordinary chat into a lesson.\n"
            "- Below you will find the COGNITIVE FIELD and RECALLED FACTS.\n"
            "- CRITICAL: Use memory naturally. DO NOT use meta-phrases like 'according to my memory', 'based on the context', or mention internal labels, scores, paths, layers, or source tags.\n"
            "- Never explain intent extraction, roots, lemmas, dictionary forms, or graph nodes to the user unless they explicitly ask about language or the system internals.\n"
            "- For short continuation messages, infer the conversational intent from recent messages instead of treating the text as a grammar exercise.\n"
            "- If the user sends a short action fragment (for example 'I'm going', 'gidiyorum') in a travel/route context, respond as if they are continuing the route conversation; do not define the verb.\n"
            "- If the user asks for a destination, blocker, owner, decision, or prior fact and [ANSWER FOCUS] has a primary_candidate, lead with that candidate in the first sentence. Do not list generic alternatives first.\n"
            "- If the user reports a current problem or situation, answer the current problem first. Long-term memory can supplement it, but lack of recalled facts must not block practical help.\n"
            "- Respect the answer_policy and confidence in [ANSWER FOCUS]: high = answer directly, medium = cautious memory-backed guess, low/no candidate = ask a concise follow-up.\n"
            "- Never invent details from concept labels, graph connections, scores, or traversal paths. Only state a remembered detail if it appears in the recalled fact text or follows directly from the user's message.\n"
            "- If the recalled facts are insufficient, say so naturally and ask a concise follow-up instead of guessing.\n"
            "- Exception: if the user explicitly asks for a guess or opinion (e.g., 'sence', 'tahmin et'), you may give one brief, clearly framed guess from the available facts, then ask at most one follow-up.\n"
            "- CRITICAL: ALWAYS respond in the EXACT SAME LANGUAGE the user is speaking.\n"
        )

    # ================================================================== #
    #  [2] GRAPH STATE — Intent topology for context resolution           #
    # ================================================================== #

    def _build_graph_state(self, recall_result):
        field = recall_result.get("cognitive_field") or {}
        lines = ["[COGNITIVE FIELD]"]

        seeds = field.get("seed_intents", [])
        activated = field.get("activated_intents", [])

        # Seeds: what the query directly hit
        if seeds:
            parts = []
            for s in seeds[:6]:
                label = s.get("label", "")
                domain = s.get("domain", "")
                energy = s.get("energy", 0.0)
                if domain and domain != "general":
                    parts.append(f"{label} [{domain}] (energy:{energy:.2f})")
                else:
                    parts.append(f"{label} (energy:{energy:.2f})")
            lines.append(f"Query hit: {', '.join(parts)}")

        # Resonant: what activated through edges
        resonant = [a for a in activated if a.get("role") != "seed" and a.get("label")]
        if resonant:
            parts = []
            for r in resonant[:6]:
                parts.append(f"{r['label']}({r.get('energy', 0.0):.2f})")
            lines.append(f"Resonant: {', '.join(parts)}")

        # Edges: THE critical piece
        all_items = seeds + resonant
        edges_seen = set()
        edge_lines = []
        for item in all_items:
            label = item.get("label", "")
            for edge in item.get("edges", [])[:4]:
                target = edge.get("target", "")
                pair = tuple(sorted([label, target]))
                if target and target != label and pair not in edges_seen:
                    edges_seen.add(pair)
                    w = edge.get("weight", 0)
                    c = edge.get("confidence", 0)
                    edge_lines.append(f"  {label} ↔ {target} (weight:{w:.2f}, confidence:{c:.2f})")

        if edge_lines:
            lines.append("\nConnections:")
            lines.extend(edge_lines[:12])

        # Traversal paths from recall (how we got from query to memories)
        paths = set()
        for bucket in ["direct_memories", "associated_memories", "weak_echo_memories"]:
            for item in recall_result.get(bucket, []):
                path = item.get("path", [])
                if len(path) > 1:
                    paths.add(" → ".join(path))
        
        if paths:
            lines.append("\nTraversal paths:")
            for p in list(paths)[:6]:
                lines.append(f"  {p}")

        if len(lines) == 1:
            lines.append("No active graph connections.")

        return "\n".join(lines)

    # ================================================================== #
    #  [3] MEMORIES — Grouped by topic, with edge provenance              #
    # ================================================================== #

    def _build_memories(self, recall_result):
        memories = []
        for bucket in ["direct_memories", "associated_memories", "weak_echo_memories"]:
            for item in recall_result.get(bucket, []):
                memories.append(item)

        memories = sorted(memories, key=lambda x: x["score"], reverse=True)[:20]

        if not memories:
            return "=== RECALLED FACTS ===\nNo specific facts recalled. Do not pretend to remember details."

        lines = [
            "=== RECALLED FACTS ===",
            "Use these candidate memories in order of relevance. Direct facts are strongest; associated facts are supporting context; weak echoes are only hints.",
        ]
        seen_chunks = set()
        index = 1
        for item in memories:
            chunk = item["chunk"]
            chunk_id = getattr(chunk, "chunk_id", None)
            if chunk_id and chunk_id in seen_chunks:
                continue
            if chunk_id:
                seen_chunks.add(chunk_id)

            layer = item.get("layer", 0)
            layer_name = {0: "direct", 1: "associated", 2: "associated", 3: "weak_echo"}.get(layer, "associated")

            raw_text = chunk.text
            if layer == 0:
                text = self._compress(raw_text, max_len=600)
            elif layer in (1, 2):
                text = self._compress(raw_text, max_len=220)
            else:
                text = self._compress(raw_text, max_len=120)

            if not text:
                continue

            lines.append(f'{index}. ({layer_name}) "{text}"')
            index += 1

        return "\n".join(lines)

    # ================================================================== #
    #  UTILS                                                              #
    # ================================================================== #

    def _compress(self, text: str, max_len: int = 400) -> str:
        text = " ".join(text.split())
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."
