import os
import sys
import json
import time
from pathlib import Path
import uvicorn
from typing import List, Dict, Any, Optional
from collections import OrderedDict
import threading
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SRC = PROJECT_ROOT / "src"
if LOCAL_SRC.exists():
    sys.path.insert(0, str(LOCAL_SRC))

from intentmind.runtime import IntentmindMemory
from intentmind.embeddings import SentenceTransformerEmbedder

load_dotenv()

app = FastAPI(title="Cognitive Intent Runtime API")

cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "INTENTMIND_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

allow_all = "*" in cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if not allow_all else ["*"],
    allow_credentials=not allow_all,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[Sistem] Embedder Yükleniyor...")
embedder = SentenceTransformerEmbedder()

# ── Per-session memory isolation ────────────────────────────────────────────
# The public demo must never mix one visitor's memories into another's, so each
# browser session gets its own fresh cognitive graph. Nothing is persisted to
# disk; sessions live in RAM only and the oldest are evicted past the cap.
MAX_SESSIONS = int(os.getenv("INTENTMIND_MAX_SESSIONS", "200"))
_SESSIONS: "OrderedDict[str, IntentmindMemory]" = OrderedDict()
_SESSIONS_LOCK = threading.Lock()


def get_session_memory(session_id: str | None) -> IntentmindMemory:
    sid = (session_id or "").strip() or "default"
    with _SESSIONS_LOCK:
        mem = _SESSIONS.get(sid)
        if mem is None:
            mem = IntentmindMemory(embedder=embedder, index_type="faiss")
            _SESSIONS[sid] = mem
            while len(_SESSIONS) > MAX_SESSIONS:
                _SESSIONS.popitem(last=False)  # evict least-recently-used
        else:
            _SESSIONS.move_to_end(sid)
        return mem


def reset_session_memory(session_id: str | None) -> None:
    sid = (session_id or "").strip() or "default"
    with _SESSIONS_LOCK:
        _SESSIONS.pop(sid, None)

model_name = os.getenv("OPENAI_MODEL")
if not model_name:
    print("UYARI: OPENAI_MODEL tanımlanmamış, .env dosyanızı kontrol edin.")
    model_name = "gpt-4o-mini"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

STORE_USER_CHAT = os.getenv("INTENTMIND_STORE_USER_CHAT", "1").lower() in {"1", "true", "yes"}
STORE_ALL_USER_CHAT = os.getenv("INTENTMIND_STORE_ALL_USER_CHAT", "0").lower() in {"1", "true", "yes"}
MAX_SHORT_HISTORY = int(os.getenv("INTENTMIND_SHORT_HISTORY", "6"))
API_PORT = int(os.getenv("INTENTMIND_API_PORT", "8002"))
API_RELOAD = os.getenv("INTENTMIND_RELOAD", "0").lower() in {"1", "true", "yes"}

_EXPLICIT_MEMORY_MARKERS = (
    "remember this",
    "remember that",
    "please remember",
    "save this",
    "keep in mind",
    "bunu hatırla",
    "şunu hatırla",
    "unutma",
    "kaydet",
    "aklında tut",
)

_REQUEST_ONLY_TERMS = {
    "sence", "nereye", "nerede", "nasıl", "ne", "kim", "tahmin", "mi", "mı",
    "where", "what", "how", "who", "guess", "think", "you",
}

_EVENT_PROBLEM_MARKERS = {
    "flat", "puncture", "punctured", "broken", "broke", "missing", "lost", "stuck",
    "blocked", "failed", "error", "issue", "problem", "crash", "crashed",
    "patladı", "patladi", "bozuldu", "yok", "kayıp", "kayip", "takıldı", "takildi",
    "kaldım", "kaldim", "hata", "sorun", "engel",
}

_QUESTION_START_TERMS = {
    "what", "where", "who", "which", "why", "how",
    "ne", "nereye", "nerede", "kim", "hangi", "neden", "nasıl",
}


def is_substantive_memory_statement(text: str) -> bool:
    tokens = [tok for tok in text.casefold().replace("?", " ").replace("!", " ").split() if tok]
    if len(tokens) < 4:
        return False
    if "?" in text or tokens[0] in _QUESTION_START_TERMS:
        return False

    non_request_tokens = [tok for tok in tokens if tok not in _REQUEST_ONLY_TERMS]
    if len(non_request_tokens) < 3:
        return False

    lowered = text.casefold()
    if any(marker in lowered for marker in _EVENT_PROBLEM_MARKERS):
        return True

    # Store descriptive statements, not tiny follow-up requests.
    return "?" not in text and len(non_request_tokens) >= 5


def should_store_user_chat(text: str, mem: IntentmindMemory) -> bool:
    """
    Long-term memory should contain durable user facts, not every short query.
    Short-term chat history already handles follow-ups inside the current turn.
    """
    if not text or not text.strip():
        return False
    if STORE_ALL_USER_CHAT:
        return True

    cache = getattr(mem._intent_engine, "_grounded_cache", {})
    constraints = cache.get("constraints", []) if isinstance(cache, dict) else []
    if constraints:
        return True

    lowered = text.casefold()
    return (
        any(marker in lowered for marker in _EXPLICIT_MEMORY_MARKERS)
        or is_substantive_memory_statement(text)
    )


def build_memory_query(messages: List[ChatMessage]) -> str:
    recent = messages[-MAX_SHORT_HISTORY:]
    if len(recent) <= 1:
        return recent[-1].content if recent else ""
    
    # We format this so the IntentEngine knows exactly what the CURRENT question is,
    # and treats the rest merely as conversational background for coreference.
    lines = ["--- Conversation History ---"]
    for msg in recent[:-1]:
        role = "assistant" if msg.role == "assistant" else "user"
        lines.append(f"{role}: {msg.content}")
    lines.append("--- Current Query (EXTRACT INTENTS PRIMARILY FOR THIS) ---")
    lines.append(f"user: {recent[-1].content}")
    
    return "\n".join(lines)

def build_response_guard(user_msg: str, system_prompt: str) -> str:
    answer_focus = extract_answer_focus(system_prompt)
    primary = answer_focus.get("primary_candidate") or {}
    policy = answer_focus.get("answer_policy") or "No structured answer focus."
    focus = json.dumps({
        "request_type": answer_focus.get("request_type"),
        "answer_policy": policy,
        "current_observation": answer_focus.get("current_observation"),
        "primary_candidate": primary,
    }, ensure_ascii=False)
    return (
        "CURRENT RESPONSE MODE OVERRIDE:\n"
        "- Treat the latest user message as ordinary conversation unless it explicitly asks about grammar, dictionary form, roots, lemmas, or intent extraction.\n"
        "- Do NOT answer with a dictionary/root/lemma explanation for ordinary chat fragments such as 'gidiyorum' or 'I'm going'.\n"
        "- If the user asks for a remembered destination, blocker, owner, decision, or fact and ANSWER FOCUS has a primary_candidate, lead with it before generic alternatives.\n"
        "- Structured answer focus:\n"
        f"{focus}"
    )


def extract_answer_focus(system_prompt: str) -> dict:
    marker = "[ANSWER FOCUS]"
    start = system_prompt.find(marker)
    if start < 0:
        return {}
    payload_start = start + len(marker)
    end_candidates = [
        index for index in (
            system_prompt.find("\n[", payload_start),
            system_prompt.find("\n===", payload_start),
        )
        if index >= 0
    ]
    payload_end = min(end_candidates) if end_candidates else len(system_prompt)
    payload = system_prompt[payload_start:payload_end].strip()
    try:
        return json.loads(payload)
    except Exception:
        return {}


def build_openai_messages(system_prompt: str, messages: List[ChatMessage]) -> list[dict]:
    oai_msgs = [{"role": "system", "content": system_prompt}]
    if messages:
        oai_msgs.append({"role": "system", "content": build_response_guard(messages[-1].content, system_prompt)})
    for msg in messages[-MAX_SHORT_HISTORY:]:
        role = msg.role if msg.role in {"user", "assistant"} else "user"
        oai_msgs.append({"role": role, "content": msg.content})
    return oai_msgs


def looks_like_language_analysis_leak(user_msg: str, response: str) -> bool:
    user_lower = (user_msg or "").casefold()
    asked_language = any(token in user_lower for token in [
        "kök", "koku", "fiil", "mast", "sözlük", "sozluk", "lemma", "root", "dictionary", "intent"
    ])
    if asked_language:
        return False
    resp = (response or "").casefold()
    return (
        ("fiil" in resp and ("temel" in resp or "sözlük" in resp or "sozluk" in resp or "gitmek" in resp))
        or ("dictionary form" in resp)
        or ("root form" in resp)
        or ("lemma" in resp)
    )


def fallback_conversation_response(user_msg: str, system_prompt: str) -> str:
    answer_focus = extract_answer_focus(system_prompt)
    candidate = answer_focus.get("primary_candidate") or {}
    label = candidate.get("label")
    if label:
        return f"Bence {label} tarafı en güçlü hafıza adayı gibi duruyor. Eminlik: {candidate.get('confidence', 'medium')}."
    return "Bir yere gidiyorsun gibi okuyorum; bunu dil bilgisi sorusu değil, rota sohbetinin devamı olarak alıyorum. Nereye olduğunu tahmin etmemi istiyorsan biraz daha ipucu ver."


def build_query_graph(memory: IntentmindMemory, mem_result: dict, max_nodes: int = 35) -> tuple[list[dict], list[str]]:
    store = memory._store
    node_order: list[str] = []
    active_ids: set[str] = set()

    def add_intent_id(intent_id: str | None, active: bool = False) -> None:
        if not intent_id or intent_id not in store.intents:
            return
        if intent_id not in node_order:
            node_order.append(intent_id)
        if active:
            active_ids.add(store.intents[intent_id].label)

    def add_label(label: str | None, active: bool = False) -> None:
        if not label:
            return
        intent = store.find_intent_by_label(label)
        if intent:
            add_intent_id(intent.intent_id, active=active)

    cog_field = mem_result.get("cognitive_field", {})
    for seed in cog_field.get("seed_intents", []):
        add_intent_id(seed.get("intent_id"), active=True)

    for item in mem_result.get("memories", {}).get("items", [])[:12]:
        is_direct = item.get("layer") == 0
        item_intent_id = item.get("intent_id")
        if item_intent_id:
            add_intent_id(item_intent_id, active=is_direct)
        else:
            add_label(item.get("intent"), active=is_direct)
        for label in item.get("path", []):
            if label == item.get("intent") and item_intent_id:
                add_intent_id(item_intent_id, active=is_direct)
            else:
                add_label(label, active=is_direct and label == item.get("intent"))

    for item in cog_field.get("activated_intents", [])[:max_nodes]:
        add_intent_id(item.get("intent_id"), active=item.get("role") == "seed")

    for label in mem_result.get("extracted_query_intents", []):
        add_label(label, active=True)

    # Expand the graph to include immediate neighbors of active nodes
    expanded_relevant = set(node_order[:max_nodes])
    for intent_id in list(expanded_relevant):
        neighbors = sorted(store.get_neighbors(intent_id, include_candidates=True), key=lambda x: x[1].weight * x[1].confidence, reverse=True)
        for target_id, edge in neighbors[:25]:
            if target_id not in expanded_relevant:
                expanded_relevant.add(target_id)
                if target_id not in node_order:
                    node_order.append(target_id)

    relevant = set(node_order[:max_nodes])
    nodes = []
    for intent_id in node_order[:max_nodes]:
        intent = store.intents.get(intent_id)
        if not intent:
            continue
        edges = []
        for target_id, edge in store.get_neighbors(intent_id, include_candidates=True):
            if target_id not in relevant:
                continue
            target = store.intents.get(target_id)
            if not target:
                continue
            edges.append({
                "target": target.label,
                "target_id": target_id,
                "weight": round(edge.weight, 3),
                "confidence": round(edge.confidence, 3),
            })

        nodes.append({
            "id": intent.label,
            "intent_id": intent_id,
            "energy": round(intent.energy, 4),
            "hitCount": intent.source_count,
            "edges": edges,
            "lastUpdate": intent.last_active,
        })

    # Inject ephemeral newly extracted intents so the user sees them instantly!
    existing_labels = {n["id"] for n in nodes}
    for label in mem_result.get("extracted_query_intents", []):
        if label not in existing_labels:
            nodes.append({
                "id": label,
                "intent_id": f"temp_{label}",
                "energy": 1.0,
                "hitCount": 1,
                "edges": [],
                "lastUpdate": time.time(),
            })
            active_ids.add(label)

    return nodes, list(active_ids)


def build_stored_memory_trace(memory: IntentmindMemory, chunk_id: str | None) -> dict | None:
    if not chunk_id:
        return None
    store = memory._store
    chunk = store.chunks.get(chunk_id)
    if not chunk:
        return None

    intents = [
        store.intents[intent_id]
        for intent_id in chunk.intent_ids
        if intent_id in store.intents
    ]
    relevant_ids = {intent.intent_id for intent in intents}
    edge_items = []
    seen_edges = set()
    for intent in intents:
        for target_id, edge in store.get_neighbors(intent.intent_id, include_candidates=True):
            if target_id not in relevant_ids or edge.edge_id in seen_edges:
                continue
            target = store.intents.get(target_id)
            if not target:
                continue
            seen_edges.add(edge.edge_id)
            edge_items.append({
                "from": intent.label,
                "to": target.label,
                "type": edge.edge_type,
                "weight": round(edge.weight, 3),
                "confidence": round(edge.confidence, 3),
                "state": edge.state,
            })

    return {
        "chunk_id": chunk.chunk_id,
        "source": chunk.source,
        "text": chunk.text[:180],
        "intents": [intent.label for intent in intents],
        "edges": edge_items,
    }


def find_latest_chunk_id_by_text(memory: IntentmindMemory, text: str) -> str | None:
    if not text:
        return None
    matches = [
        chunk
        for chunk in memory._store.chunks.values()
        if chunk.text == text
    ]
    if not matches:
        return None
    matches.sort(key=lambda chunk: chunk.created_at, reverse=True)
    return matches[0].chunk_id


@app.post("/api/reset")
async def reset_session(request: Request):
    """Drop the caller's session graph so they can start from a clean brain."""
    reset_session_memory(request.headers.get("x-session-id"))
    return {"ok": True}


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    try:
        # Get the latest user message
        if not req.messages:
            return {"response": "API CRASH: empty message list", "field": {"nodes": [], "active_ids": [], "stats": {}}}

        # Each browser session owns an isolated cognitive graph.
        mem = get_session_memory(request.headers.get("x-session-id"))

        user_msg = req.messages[-1].content
        memory_query = build_memory_query(req.messages)

        # Query first. Chat text is short-term context; it should not create
        # long-term graph nodes unless explicitly enabled by env.
        mem_result = mem.query(user_query=user_msg, context_text=memory_query)
        # Use the prompt already built by runtime.query() — it has access
        # to the raw recall_result (direct_memories, associated_memories etc.)
        # which PromptBuilder needs.  Re-creating a PromptBuilder here and
        # feeding it mem_result was broken: mem_result packs memories under
        # "memories.items", but PromptBuilder expects "direct_memories" etc.
        system_prompt = mem_result["prompt"]

        # === DEBUG: Log what's actually going to the LLM ===
        print("\n" + "=" * 60)
        print("SYSTEM PROMPT SENT TO LLM:")
        print("=" * 60)
        print(system_prompt[:2000])
        if len(system_prompt) > 2000:
            print(f"... [truncated, total {len(system_prompt)} chars]")
        print("=" * 60 + "\n")

        # Short-term history resolves follow-ups ("sunuculara gönderiyor mu?")
        # while the graph supplies long-term recalled knowledge.
        oai_msgs = build_openai_messages(system_prompt, req.messages)
            
        try:
            if client is None:
                raise RuntimeError("OPENAI_API_KEY is not configured. See .env.example.")
            completion = client.chat.completions.create(
                model=model_name,
                messages=oai_msgs
            )
            ai_response = completion.choices[0].message.content
            if looks_like_language_analysis_leak(user_msg, ai_response):
                ai_response = fallback_conversation_response(user_msg, system_prompt)
        except Exception as e:
            ai_response = f"API error: {str(e)}"
        
        stored_chunk_id = None
        if STORE_USER_CHAT and should_store_user_chat(user_msg, mem):
            stored_result = mem.add(text=user_msg, source="user")
            stored_chunk_id = getattr(stored_result, "chunk_id", stored_result)
            if not stored_chunk_id:
                stored_chunk_id = find_latest_chunk_id_by_text(mem, user_msg)

        # AI response ingestion DISABLED.
        # Reason: AI's own verbose responses were flooding the memory graph,
        # getting higher recall scores than actual source data (news, facts),
        # creating a self-reinforcing loop that drowned out real knowledge.
        # To re-enable: uncomment the line below.
        # if not ai_response.startswith("API error:"):
        #     memory.add(text=ai_response, source="ai")
        
        # Tick memory decay
        mem.tick()

        # Build UI graph AFTER ingestion and decay so new nodes and edges exist!
        nodes, active_ids = build_query_graph(mem, mem_result)

        # NOTE: no disk persistence — session graphs are intentionally ephemeral
        # so the public demo stays clean for every visitor.

        stats_dict = mem._store.stats()
        all_energies = [n.energy for n in mem._store.intents.values()]
        avg_energy = sum(all_energies) / len(all_energies) if all_energies else 0.0

        # Build cognitive path for UI transparency
        cog_field = mem_result.get("cognitive_field", {})
        seeds = cog_field.get("seed_intents", [])
        resonant = [
            item for item in cog_field.get("activated_intents", [])
            if item.get("role") != "seed"
        ][:8]
        
        # Build path items from recalled memories
        path_items = []
        for item in mem_result["memories"]["items"][:8]:
            path_items.append({
                "intent": item["intent"],
                "layer": item["layer"],
                "score": round(item["score"], 3),
                "path_strength": item.get("path_strength", 0),
                "called_by": item.get("called_by"),
                "reason": item.get("reason"),
                "path": item.get("path", []),
                "source": item["source"],
                "text": item["text"][:120],
            })

        stored_trace = build_stored_memory_trace(mem, stored_chunk_id)

        return {
            "response": ai_response,
            "field": {
                "nodes": nodes,
                "active_ids": active_ids,
                "stats": {
                    "total_nodes": stats_dict.get("intents", 0),
                    "total_edges": stats_dict.get("edges", 0),
                    "avg_energy": avg_energy,
                    "total_chunks": stats_dict.get("chunks", 0)
                }
            },
            "cognitive_path": {
                "seeds": seeds,
                "resonant": resonant,
                "memories_used": len(path_items),
                "items": path_items,
                "newly_stored": stored_trace,
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        fallback_stats = {"total_nodes": 0, "total_edges": 0, "avg_energy": 0.0, "total_chunks": 0}
        return {"response": f"API CRASH: {str(e)}", "field": {"nodes": [], "active_ids": [], "stats": fallback_stats}}

ui_dist = PROJECT_ROOT / "ui" / "dist"
if ui_dist.exists():
    app.mount("/assets", StaticFiles(directory=ui_dist / "assets"), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_ui(full_path: str):
        if full_path.startswith("api/"):
            return {"error": "Not found"}
        file_path = ui_dist / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(ui_dist / "index.html")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=API_PORT, reload=API_RELOAD)
