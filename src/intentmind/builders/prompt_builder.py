from __future__ import annotations

from collections import defaultdict


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

    # ================================================================== #
    #  PUBLIC API                                                         #
    # ================================================================== #

    def build(self, user_query, recall_result, cognitive_state):
        role = self._build_role()
        graph = self._build_graph_state(recall_result)
        memories = self._build_memories(recall_result)
        
        prompt = f"{role}\n\n{graph}\n\n{memories}".strip()
        
        if len(prompt) <= self.max_chars:
            return prompt
        return prompt[:self.max_chars] + "\n[TRIMMED]"

    # ================================================================== #
    #  [1] ROLE — 3 lines, zero waste                                     #
    # ================================================================== #

    def _build_role(self):
        return (
            "You are an intelligent, conversational assistant.\n"
            "RULES:\n"
            "- You have been provided with some RECALLED FACTS below.\n"
            "- CRITICAL: Speak as if you already know these facts naturally. DO NOT ever say 'according to the recalled facts', 'based on the context', 'my memories say', or mention any source tags like '[news]'.\n"
            "- Synthesize the facts directly into your answer.\n"
            "- If the facts answer the user's question, rely on them heavily but weave them into a natural response.\n"
            "- If the facts are insufficient, supplement with your general knowledge naturally.\n"
            "- Respond in the user's language. Be direct, confident, and conversational."
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
            return "=== RECALLED FACTS ===\nNo specific facts recalled. Use general knowledge."

        # Group by intent topic
        groups = defaultdict(list)
        for item in memories:
            intent = item["intent"]
            label = intent.label if hasattr(intent, "label") else str(intent)
            domain = getattr(intent, "domain", "general")
            score = round(item["score"], 2)
            layer = item.get("layer", 0)
            
            # Cognitive Token Budgeting: Length depends on the layer (Ring)
            raw_text = item["chunk"].text
            if layer == 0:
                # Ring 0 / Direct: High budget
                text = self._compress(raw_text, max_len=600)
            elif layer == 1:
                # Ring 1 / Strong Neighbors: Medium budget
                text = self._compress(raw_text, max_len=200)
            elif layer == 2:
                # Ring 2 / Weak Associative: Low budget (just facts/summaries)
                text = self._compress(raw_text, max_len=80)
            else:
                # Ring 3 / Echo: Signal only
                text = ""

            called_by = item.get("called_by")
            edge_type = item.get("edge_type", "")
            reason = item.get("reason", "")
            source = item["chunk"].source
            
            # Build provenance string
            via = ""
            if called_by:
                via = f" (via: {called_by}"
                if edge_type:
                    via += f", edge: {edge_type}"
                via += ")"
            
            layer_name = {0: "direct", 1: "associated", 2: "weak_echo", 3: "faint"}.get(layer, str(layer))
            
            groups[label].append({
                "score": score,
                "text": text,
                "layer": layer_name,
                "via": via,
                "source": source,
                "domain": domain,
            })

        lines = ["=== RECALLED FACTS ==="]
        for topic, items in groups.items():
            domain = items[0]["domain"] if items[0]["domain"] != "general" else ""
            domain_str = f" [{domain}]" if domain else ""
            lines.append(f"\n▸ {topic}{domain_str}:")
            for m in items[:5]:
                source_tag = f"[{m['source']}]" if m['source'] != 'chat' else ""
                layer_tag = f"[{m['layer']}, {m['score']}]"
                
                # If Ring 3, we don't output text, just the signal.
                if m["text"]:
                    lines.append(f'  {layer_tag}{m["via"]}{" " + source_tag if source_tag else ""} "{m["text"]}"')
                else:
                    lines.append(f'  {layer_tag}{m["via"]}{" " + source_tag if source_tag else ""} (activation signal only)')

        return "\n".join(lines)

    # ================================================================== #
    #  UTILS                                                              #
    # ================================================================== #

    def _compress(self, text: str, max_len: int = 400) -> str:
        text = " ".join(text.split())
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."
