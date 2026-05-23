import json
from typing import Any, Callable, Dict


def load_fixture(json_path: str) -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_fixture_extractor(data: Dict[str, Any]) -> Callable[[str], list[str]]:
    """
    Build a deterministic stand-in for a faithful LLM extractor.

    This is for benchmark repeatability. It lets us evaluate graph/recall
    behavior without depending on live LLM calls or hand-tuned FakeEmbedder
    vocabulary.
    """
    by_text = {}
    for item in data.get("memories", []):
        intents = item.get("intents", [])
        if intents:
            by_text[item["text"]] = intents
    for item in data.get("queries", []):
        intents = item.get("expected_intents", [])
        if intents:
            by_text[item["query"]] = intents

    def extractor(text: str) -> list[str]:
        return list(by_text.get(text, []))

    return extractor

def load_fixture_and_map(memory_instance, json_path: str) -> Dict[str, Any]:
    """
    Loads a benchmark fixture JSON, ingests memories into the provided IntentmindMemory instance,
    and returns the mapped queries taking automatic deduplication into account.
    """
    data = load_fixture(json_path)

    chunk_mapping = {}
    for item in data.get("memories", []):
        surviving_id = memory_instance.add(
            text=item["text"], 
            source="benchmark", 
            chunk_id=item["chunk_id"]
        )
        chunk_mapping[item["chunk_id"]] = surviving_id

    # Update expected queries based on merges
    for q in data.get("queries", []):
        if "expected_chunks" in q:
            mapped = set(chunk_mapping.get(cid, cid) for cid in q["expected_chunks"])
            q["expected_chunks"] = list(mapped)
            
    return data
