from __future__ import annotations

from ..models import CognitiveState
from ..embeddings import BaseEmbedder, cosine_similarity


class CognitiveStateEngine:
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self.prototypes = {
            "neutral":    embedder.embed("I am providing or requesting information."),
            "curious":   embedder.embed("I am curious and want to explore this idea."),
            "urgent": embedder.embed("This is critical, urgent, and needs immediate attention."),
            "confident":   embedder.embed("This makes sense and is solid."),
            "skeptical":   embedder.embed("I am unsure and this needs verification."),
            "cautious":   embedder.embed("This is risky and requires careful attention."),
        }
        self.override_priority = ["urgent", "cautious", "skeptical", "curious", "confident", "neutral"]
        self.override_threshold = 0.80

    def detect(self, message_embedding) -> CognitiveState:
        scores = {emotion: cosine_similarity(message_embedding, proto) for emotion, proto in self.prototypes.items()}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_emotion, top_score = ranked[0]
        second_score = ranked[1][1]
        delta = top_score - second_score
        for priority_emotion in self.override_priority[:2]:
            if scores[priority_emotion] > self.override_threshold:
                top_emotion = priority_emotion
                top_score = scores[priority_emotion]
                break
        if top_score < 0.55 or delta < 0.07:
            state = "neutral"
        else:
            state = top_emotion
        return CognitiveState(current=state, confidence=round(top_score, 3), weak_echo=state in ["curious", "urgent"])
