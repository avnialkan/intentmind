import os
import sys
import json
import time
from pathlib import Path
import uvicorn
from typing import List, Dict, Any, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("[Sistem] Embedder ve Hafıza Yükleniyor...")
embedder = SentenceTransformerEmbedder()

db_path = "news_memory.json"
if os.path.exists(db_path):
    print(f"[Sistem] {db_path} yükleniyor...")
    memory = IntentmindMemory.load(db_path, embedder=embedder)
else:
    memory = IntentmindMemory(embedder=embedder, index_type="faiss")

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
MAX_SHORT_HISTORY = int(os.getenv("INTENTMIND_SHORT_HISTORY", "6"))


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

def build_openai_messages(system_prompt: str, messages: List[ChatMessage]) -> list[dict]:
    oai_msgs = [{"role": "system", "content": system_prompt}]
    for msg in messages[-MAX_SHORT_HISTORY:]:
        role = msg.role if msg.role in {"user", "assistant"} else "user"
        oai_msgs.append({"role": role, "content": msg.content})
    return oai_msgs


def build_query_graph(memory: IntentmindMemory, mem_result: dict, max_nodes: int = 20) -> tuple[list[dict], list[str]]:
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
        add_label(item.get("intent"), active=is_direct)
        for label in item.get("path", []):
            add_label(label, active=is_direct and label == item.get("intent"))

    if not node_order:
        for item in cog_field.get("activated_intents", [])[:max_nodes]:
            add_intent_id(item.get("intent_id"), active=item.get("role") == "seed")

    relevant = set(node_order[:max_nodes])
    nodes = []
    for intent_id in node_order[:max_nodes]:
        intent = store.intents.get(intent_id)
        if not intent:
            continue
        edges = []
        for target_id, edge in store.get_neighbors(intent_id):
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


@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        # Get the latest user message
        if not req.messages:
            return {"response": "API CRASH: empty message list", "field": {"nodes": [], "active_ids": [], "stats": {}}}
        user_msg = req.messages[-1].content
        memory_query = build_memory_query(req.messages)

        # Query first. Chat text is short-term context; it should not create
        # long-term graph nodes unless explicitly enabled by env.
        mem_result = memory.query(memory_query)
        nodes, active_ids = build_query_graph(memory, mem_result)

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
        except Exception as e:
            ai_response = f"API error: {str(e)}"
        
        if STORE_USER_CHAT and user_msg.strip():
            memory.add(text=user_msg, source="user")

        # AI response ingestion DISABLED.
        # Reason: AI's own verbose responses were flooding the memory graph,
        # getting higher recall scores than actual source data (news, facts),
        # creating a self-reinforcing loop that drowned out real knowledge.
        # To re-enable: uncomment the line below.
        # if not ai_response.startswith("API error:"):
        #     memory.add(text=ai_response, source="ai")
        
        # Tick memory decay
        memory.tick()
        
        stats_dict = memory._store.stats()
        all_energies = [n.energy for n in memory._store.intents.values()]
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
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        fallback_stats = {"total_nodes": 0, "total_edges": 0, "avg_energy": 0.0, "total_chunks": 0}
        return {"response": f"API CRASH: {str(e)}", "field": {"nodes": [], "active_ids": [], "stats": fallback_stats}}

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
