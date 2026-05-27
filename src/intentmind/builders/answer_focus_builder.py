from __future__ import annotations

import json
import re
from collections import defaultdict


class AnswerFocusBuilder:
    """
    Converts awakened graph memories into an answer-oriented evidence frame.

    The graph can wake concepts, but the assistant should answer from evidence.
    This builder gives the response model a compact, domain-neutral view of:
    request type, likely answer candidates, confidence, and supporting text.
    """

    _REQUEST_HINTS = {
        "destination": {
            "where", "route", "destination", "going", "went",
            "nereye", "nerede", "gidiyorum", "giderdim", "gidecektim",
        },
        "blocker": {
            "blocked", "blocking", "blocker", "stuck", "risk", "issue",
            "engel", "takıldı", "risk", "sorun",
        },
        "owner": {"who", "owner", "responsible", "kim", "sahip", "sorumlu"},
        "decision": {"decide", "decision", "karar", "seçim", "chosen"},
    }

    _EDGE_TO_TYPE = {
        "spatial_link": "destination_or_location",
        "instrumental_link": "method_or_tool",
        "causal_link": "cause_or_blocker",
        "event_link": "event",
        "actor_link": "person_or_owner",
        "possessive_link": "person_or_owner",
        "thematic_link": "related_context",
        "temporal_link": "related_context",
        "co_occurrence_link": "related_context",
    }

    def build(self, user_query: str, recall_result: dict) -> dict:
        request_type = self._infer_request_type(user_query, recall_result)
        current_observation = self._current_observation(user_query, request_type)
        evidence = self._collect_evidence(recall_result)
        candidates = self._rank_candidates(evidence, request_type)
        top = candidates[0] if candidates else None

        return {
            "request_type": request_type,
            "answer_policy": self._answer_policy(request_type, top),
            "current_observation": current_observation,
            "primary_candidate": top,
            "supporting_evidence": self._supporting_evidence(evidence, request_type),
        }

    def build_block(self, user_query: str, recall_result: dict) -> str:
        focus = self.build(user_query, recall_result)
        return "[ANSWER FOCUS]\n" + json.dumps(focus, ensure_ascii=False, indent=2)

    def _infer_request_type(self, user_query: str, recall_result: dict) -> str:
        request_desc = (recall_result.get("request_description") or "").casefold()
        query = (user_query or "").casefold()

        # Live problem reports must win over generic words in the extractor's
        # request summary. Example: "where the user's tire is flat" contains
        # "where", but the user is not asking for a destination.
        if self._looks_like_current_problem(query):
            return "current_problem"
        if self._looks_like_current_plan(query, request_desc):
            return "current_plan"

        haystack_tokens = self._tokens(f"{query} {request_desc}")
        for request_type, hints in self._REQUEST_HINTS.items():
            if any(hint.casefold() in haystack_tokens for hint in hints):
                return request_type
        if any(item.get("edge_type") == "spatial_link" for item in self._all_items(recall_result)):
            return "destination"
        return "general_recall"

    def _current_observation(self, user_query: str, request_type: str) -> dict:
        text = " ".join((user_query or "").split())
        if not text:
            return {}
        return {
            "type": request_type,
            "text": text[:300],
            "use_as_current_turn_evidence": True,
            "policy": "Use this as the live user situation. Do not confuse it with recalled long-term memory.",
        }

    def _looks_like_current_problem(self, query: str) -> bool:
        if not query:
            return False
        if self._looks_like_question(query):
            return False
        problem_markers = {
            "broken", "flat", "missing", "lost", "stuck", "blocked", "failed", "error",
            "patladı", "yok", "kayboldu", "bozuldu", "kaldım", "takıldı", "hata",
        }
        return any(marker in query for marker in problem_markers)

    def _looks_like_current_plan(self, query: str, request_desc: str = "") -> bool:
        if not query or self._looks_like_question(query):
            return False
        haystack = f"{query} {request_desc}"
        plan_markers = {
            "will", "plan", "planning", "going", "gonna",
            "yapacağım", "yapacagim", "gideceğim", "gidecegim",
            "edeceğim", "edecegim", "düşünüyorum", "dusunuyorum",
            "planlıyorum", "planliyorum", "tavsiye", "plan",
        }
        return any(marker in haystack for marker in plan_markers)

    def _looks_like_question(self, query: str) -> bool:
        if "?" in query:
            return True
        question_tokens = {
            "what", "where", "who", "which", "why", "how",
            "ne", "nereye", "nerede", "kim", "hangi", "neden", "nasıl",
        }
        return bool(self._tokens(query).intersection(question_tokens))

    def _tokens(self, text: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"\w+", text or "", flags=re.UNICODE)
            if token
        }

    def _collect_evidence(self, recall_result: dict) -> list[dict]:
        evidence = []
        seen = set()
        for item in self._all_items(recall_result):
            chunk = item.get("chunk")
            intent = item.get("intent")
            if not chunk or not intent:
                continue

            text = " ".join((chunk.text or "").split())
            if not text:
                continue
            key = (getattr(chunk, "chunk_id", ""), getattr(intent, "intent_id", ""))
            if key in seen:
                continue
            seen.add(key)

            edge_type = item.get("edge_type")
            intent_type = getattr(intent, "type", "concept")
            domain = getattr(intent, "domain", "general")
            evidence_type = self._evidence_type(edge_type, intent_type, domain)

            evidence.append({
                "type": evidence_type,
                "label": getattr(intent, "label", ""),
                "text": text[:240],
                "strength": round(float(item.get("score", 0.0)), 3),
                "support": self._support_level(item),
                "layer": item.get("layer", 0),
            })

        return sorted(
            evidence,
            key=lambda item: (
                self._type_priority(item["type"]),
                item["support"] == "direct",
                item["strength"],
            ),
            reverse=True,
        )

    def _rank_candidates(self, evidence: list[dict], request_type: str) -> list[dict]:
        if request_type in {"current_problem", "current_plan", "general_recall"}:
            return []

        buckets = defaultdict(lambda: {"score": 0.0, "evidence": []})
        desired_types = self._desired_evidence_types(request_type)

        for item in evidence:
            if desired_types and item["type"] not in desired_types:
                continue
            label = item["label"] or item["text"]
            bucket = buckets[label]
            bucket["score"] += item["strength"] * self._support_multiplier(item["support"])
            bucket["evidence"].append(item["text"])

        candidates = []
        for label, data in buckets.items():
            candidates.append({
                "label": label,
                "confidence": self._confidence(data["score"], data["evidence"]),
                "evidence": list(dict.fromkeys(data["evidence"]))[:4],
            })
        return sorted(candidates, key=lambda item: self._confidence_rank(item["confidence"]), reverse=True)

    def _answer_policy(self, request_type: str, candidate: dict | None) -> str:
        if request_type == "current_problem":
            return "Answer the live situation directly using current_observation and general practical knowledge. Ask one concise follow-up if needed."
        if request_type == "current_plan":
            return "Answer the live plan directly using current_observation. Use direct recalled facts as helpful context, but do not lead with generic associated memories."
        if not candidate:
            return "No strong evidence was awakened. Do not invent; ask one concise follow-up."
        if candidate["confidence"] == "high":
            return "Lead with the primary candidate as a remembered answer, then cite the evidence naturally."
        if request_type in {"destination", "blocker", "owner", "decision"}:
            return "Lead with the primary candidate as the best memory-backed guess, using cautious wording."
        return "Use the primary candidate only if it directly answers the user; otherwise ask a follow-up."

    def _all_items(self, recall_result: dict) -> list[dict]:
        items = []
        for bucket in ("direct_memories", "associated_memories", "weak_echo_memories"):
            items.extend(recall_result.get(bucket, []))
        return items

    def _evidence_type(self, edge_type: str | None, intent_type: str, domain: str) -> str:
        if edge_type in self._EDGE_TO_TYPE:
            return self._EDGE_TO_TYPE[edge_type]
        if intent_type in {"entity", "entity/location"}:
            return "entity"
        if intent_type == "fact":
            return "fact"
        if domain in {"Geography", "Travel"}:
            return "destination_or_location"
        if domain in {"Insurance", "Risk", "Security"}:
            return "cause_or_blocker"
        return "related_context"

    def _support_level(self, item: dict) -> str:
        layer = item.get("layer", 0)
        if layer == 0:
            return "direct"
        if layer in (1, 2):
            return "associated"
        return "weak"

    def _desired_evidence_types(self, request_type: str) -> set[str]:
        if request_type == "destination":
            return {"destination_or_location", "entity"}
        if request_type == "blocker":
            return {"cause_or_blocker", "fact"}
        if request_type == "owner":
            return {"person_or_owner", "entity"}
        if request_type == "decision":
            return {"fact", "event", "related_context"}
        return set()

    def _supporting_evidence(self, evidence: list[dict], request_type: str) -> list[dict]:
        if request_type in {"current_problem", "current_plan"}:
            direct = [item for item in evidence if item["support"] == "direct"]
            return direct[:10]
        return evidence[:10]

    def _support_multiplier(self, support: str) -> float:
        return {"direct": 1.0, "associated": 0.78, "weak": 0.35}.get(support, 0.5)

    def _type_priority(self, evidence_type: str) -> int:
        priorities = {
            "destination_or_location": 6,
            "cause_or_blocker": 6,
            "person_or_owner": 6,
            "fact": 5,
            "event": 4,
            "method_or_tool": 3,
            "entity": 3,
            "related_context": 1,
        }
        return priorities.get(evidence_type, 0)

    def _confidence(self, score: float, evidence: list[str]) -> str:
        if score >= 0.90 and len(evidence) >= 2:
            return "high"
        if score >= 0.32:
            return "medium"
        return "low"

    def _confidence_rank(self, confidence: str) -> int:
        return {"high": 3, "medium": 2, "low": 1}.get(confidence, 0)
