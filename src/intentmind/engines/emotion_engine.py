from __future__ import annotations

from ..models import EmotionalState
from ..embeddings import BaseEmbedder, cosine_similarity


class EmotionEngine:
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        self.prototypes = {
            "nötr":    embedder.embed("bilgi istiyorum"),
            "merak":   embedder.embed("bunu merak ediyorum keşfetmek istiyorum"),
            "heyecan": embedder.embed("bu fikir beni heyecanlandırıyor"),
            "güven":   embedder.embed("bu mantıklı sağlam görünüyor"),
            "şüphe":   embedder.embed("emin değilim doğrulanması gerekiyor"),
            "korku":   embedder.embed("riskli olabilir dikkatli olmak lazım"),
        }
        self.override_priority = ["korku", "şüphe", "heyecan", "merak", "güven", "nötr"]
        self.override_threshold = 0.80

    def detect(self, message_embedding) -> EmotionalState:
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
            emotion = "nötr"
        else:
            emotion = top_emotion
        return EmotionalState(current=emotion, confidence=round(top_score, 3), weak_echo=emotion in ["merak", "heyecan"])
