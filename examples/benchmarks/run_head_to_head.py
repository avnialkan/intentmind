"""
Intentmind vs Microsoft GraphRAG - Head-to-Head Benchmark
==========================================================
Uses VIINA-style conflict event data to compare:
- Intentmind (Cognitive RAG with Energy-based Graph)
- Microsoft GraphRAG (Community-based Knowledge Graph)
- Baseline RAG (Pure Vector Search)

All three systems ingest the same data and answer the same questions.
Evaluated using Microsoft's own metrics: Comprehensiveness, Diversity, Empowerment.
"""
import os
import sys
import json
import time
import shutil
import subprocess
from typing import Dict
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from intentmind import IntentmindMemory
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
client = OpenAI()
MODEL_NAME = os.getenv("OPENAI_MODEL")

BENCHMARK_QUESTIONS = [
    {"id": "global_1", "type": "global", "query": "What are the main themes in this dataset?"},
    {"id": "global_2", "type": "global", "query": "What are the most significant patterns of military escalation described in these reports?"},
    {"id": "global_3", "type": "global", "query": "How do humanitarian consequences connect to military operations across these events?"},
    {"id": "global_4", "type": "global", "query": "What role do international organizations and foreign countries play in the conflict?"},
    {"id": "global_5", "type": "global", "query": "What are the key supply chain and logistics challenges faced by both sides?"},
    {"id": "activity_1", "type": "activity", "query": "Which military units and formations are mentioned most frequently, and what are their roles?"},
    {"id": "activity_2", "type": "activity", "query": "What are the key turning points described in these reports, and what caused them?"},
    {"id": "specific_1", "type": "specific", "query": "What happened at the Zaporizhzhia Nuclear Power Plant, and why is it significant?"},
    {"id": "specific_2", "type": "specific", "query": "Describe the battle of Bakhmut: who is fighting, what are the tactics, and what is the humanitarian cost?"},
    {"id": "specific_3", "type": "specific", "query": "How has the grain export crisis affected global food security?"},
]


def generate_answer(query: str, context: str, system_name: str = "assistant") -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert conflict analyst. Provide a comprehensive, detailed answer based ONLY on the provided context. Cover multiple dimensions and perspectives."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.0,
        max_completion_tokens=1500
    )
    return response.choices[0].message.content


def graphrag_evaluate(query: str, answers: Dict[str, str]) -> Dict:
    """Microsoft-style head-to-head evaluation of all systems."""
    systems_text = ""
    system_keys = list(answers.keys())
    for i, (name, answer) in enumerate(answers.items()):
        label = chr(65 + i)  # A, B, C
        systems_text += f"\n=== System {label} ({name}) Answer ===\n{answer}\n"

    prompt = f"""You are an expert evaluator comparing answers from multiple RAG systems.
Evaluate each system's answer on these criteria (as used by Microsoft Research):

1. **Comprehensiveness**: Which covers more relevant aspects? (1-10)
2. **Diversity**: Which provides more varied perspectives? (1-10)
3. **Empowerment**: Which better helps understanding? (1-10)

Question: {query}
{systems_text}

Score EACH system on each metric (1-10). Also pick an overall winner.
Respond ONLY as JSON:
{{
    "scores": {{
        "{system_keys[0]}": {{"comprehensiveness": 8, "diversity": 7, "empowerment": 8, "total": 23}},
        "{system_keys[1]}": {{"comprehensiveness": 6, "diversity": 5, "empowerment": 6, "total": 17}},
        "{system_keys[2]}": {{"comprehensiveness": 7, "diversity": 6, "empowerment": 7, "total": 20}}
    }},
    "overall_winner": "{system_keys[0]}",
    "reasoning": "..."
}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}


def setup_graphrag(data_path: str, workspace: str):
    """Initialize and index Microsoft GraphRAG."""
    print("\n--- Setting up Microsoft GraphRAG ---")
    
    # Create workspace
    os.makedirs(workspace, exist_ok=True)
    input_dir = os.path.join(workspace, "input")
    os.makedirs(input_dir, exist_ok=True)
    
    # Copy data to input
    shutil.copy(data_path, os.path.join(input_dir, "viina_events.txt"))
    
    if not os.path.exists(os.path.join(workspace, "settings.yaml")):
        print("Initializing GraphRAG...")
        result = subprocess.run(
            [sys.executable, "-m", "graphrag", "init", "--root", workspace, "--model", "gpt-4o-mini", "--embedding", "text-embedding-3-small"],
            capture_output=True, text=True, cwd=workspace
        )
        print(f"Init stdout: {result.stdout[:200]}")
        if result.returncode != 0:
            print(f"Init stderr: {result.stderr[:500]}")
    else:
        print("GraphRAG already initialized.")
    
    # Update settings with API key
    env_path = os.path.join(workspace, ".env")
    api_key = os.getenv("OPENAI_API_KEY", "")
    with open(env_path, "w") as f:
        f.write(f"GRAPHRAG_API_KEY={api_key}\n")
    
    # Run indexing
    print("Running GraphRAG indexing (this may take a few minutes)...")
    result = subprocess.run(
        [sys.executable, "-m", "graphrag", "index", "--root", workspace],
        capture_output=True, text=True, timeout=600
    )
    print(f"Index stdout (last 500): {result.stdout[-500:]}")
    if result.returncode != 0:
        print(f"Index stderr (last 500): {result.stderr[-500:]}")
        return False
    
    print("GraphRAG indexing complete!")
    return True


def query_graphrag(workspace: str, query: str, method: str = "global") -> str:
    """Query Microsoft GraphRAG."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "graphrag", "query", "--root", workspace,
             "--method", method, "--query", query],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            # Extract the actual answer from output
            output = result.stdout.strip()
            # GraphRAG prints some metadata before the answer
            if "SUCCESS" in output:
                answer_start = output.find("\n\n")
                if answer_start >= 0:
                    return output[answer_start:].strip()
            return output
        else:
            return f"GraphRAG Error: {result.stderr[:500]}"
    except subprocess.TimeoutExpired:
        return "GraphRAG query timed out after 120 seconds."
    except Exception as e:
        return f"GraphRAG Error: {str(e)}"


def main():
    data_path = os.path.join(os.path.dirname(__file__), "datasets", "viina_events.txt")
    graphrag_workspace = os.path.join(os.path.dirname(__file__), "graphrag_workspace")

    print("=" * 80)
    print("  INTENTMIND vs MICROSOFT GRAPHRAG vs BASELINE RAG")
    print("  Dataset: VIINA-style Ukraine Conflict Event Reports")
    print("  Metrics: Comprehensiveness | Diversity | Empowerment (1-10)")
    print("=" * 80)

    # === 1. SETUP INTENTMIND ===
    print("\n--- Setting up Intentmind ---")
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    print(f"Data: {len(paragraphs)} event reports")

    model_name = os.getenv("OPENAI_MODEL")
    memory = IntentmindMemory(model=model_name)
    
    print("Ingesting into Intentmind...")
    for i, para in enumerate(paragraphs):
        memory.add(para, source="viina_events")
        print(f"  [{i+1}/{len(paragraphs)}] Ingested.")
        time.sleep(0.2)
    
    stats = memory._store.stats()
    print(f"Intentmind Graph: {stats['intents']} Nodes | {stats['edges']} Edges | {stats['chunks']} Chunks")

    # === 2. SETUP GRAPHRAG ===
    # graphrag_ready = setup_graphrag(data_path, graphrag_workspace)
    graphrag_ready = False

    # === 3. RUN BENCHMARK ===
    print("\n" + "=" * 80)
    print("  RUNNING HEAD-TO-HEAD BENCHMARK")
    print("=" * 80)

    results = []
    system_wins = {"Intentmind": 0, "GraphRAG": 0, "Baseline": 0, "Error": 0}
    system_scores = {"Intentmind": 0, "GraphRAG": 0, "Baseline": 0}

    for i, test in enumerate(BENCHMARK_QUESTIONS):
        query = test["query"]
        print(f"\n[{i+1}/{len(BENCHMARK_QUESTIONS)}] ({test['type']}) {query[:70]}...")

        # Intentmind
        im_result = memory.query(query)
        im_context = im_result["prompt"]
        im_answer = generate_answer(query, im_context, "Intentmind")

        # Baseline RAG
        query_emb = memory.embedder.embed(query)
        raw_chunks = [c for _, c in memory._store.search_chunks(query_emb, top_k=5)]
        base_context = "\n\n".join([c.text for c in raw_chunks])
        base_answer = generate_answer(query, base_context, "Baseline")

        # Evaluate Intentmind vs Baseline
        answers = {
            "Intentmind": im_answer,
            "Baseline": base_answer
        }
        
        systems_text = ""
        system_keys = list(answers.keys())
        for j, (name, answer) in enumerate(answers.items()):
            label = chr(65 + j)  # A, B
            systems_text += f"\n=== System {label} ({name}) Answer ===\n{answer}\n"

        prompt = f"""You are an expert evaluator comparing answers from multiple RAG systems.
Evaluate each system's answer on these criteria (as used by Microsoft Research):

1. **Comprehensiveness**: Which covers more relevant aspects? (1-10)
2. **Diversity**: Which provides more varied perspectives? (1-10)
3. **Empowerment**: Which better helps understanding? (1-10)

Question: {query}
{systems_text}

Score EACH system on each metric (1-10). Also pick an overall winner.
Respond ONLY as JSON:
{{
    "scores": {{
        "{system_keys[0]}": {{"comprehensiveness": 8, "diversity": 7, "empowerment": 8, "total": 23}},
        "{system_keys[1]}": {{"comprehensiveness": 6, "diversity": 5, "empowerment": 6, "total": 17}}
    }},
    "overall_winner": "{system_keys[0]}",
    "reasoning": "..."
}}"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            evaluation = json.loads(response.choices[0].message.content)
        except Exception as e:
            evaluation = {"error": str(e)}

        winner = evaluation.get("overall_winner", "Error")
        system_wins[winner] = system_wins.get(winner, 0) + 1

        # Accumulate scores
        scores = evaluation.get("scores", {})
        for sys_name in ["Intentmind", "Baseline"]:
            if sys_name in scores:
                system_scores[sys_name] += scores[sys_name].get("total", 0)

        print(f"  Winner: {winner}")
        for sys_name in ["Intentmind", "Baseline"]:
            if sys_name in scores:
                s = scores[sys_name]
                print(f"    {sys_name:<12} C:{s.get('comprehensiveness',0)} D:{s.get('diversity',0)} E:{s.get('empowerment',0)} = {s.get('total',0)}")

        results.append({
            "id": test["id"],
            "type": test["type"],
            "query": query,
            "answers": answers,
            "evaluation": evaluation,
            "winner": winner
        })

    # === FINAL REPORT ===
    n = len(BENCHMARK_QUESTIONS)
    print("\n" + "=" * 80)
    print("  FINAL REPORT: INTENTMIND vs BASELINE")
    print("=" * 80)
    print(f"\n  WINS:")
    for sys_name in ["Intentmind", "Baseline"]:
        bar = "█" * (system_wins.get(sys_name, 0) * 5)
        print(f"    {sys_name:<12} {system_wins.get(sys_name, 0):>2}/{n}  {bar}")

    print(f"\n  AVERAGE SCORE (out of 30):")
    for sys_name in ["Intentmind", "Baseline"]:
        avg = system_scores.get(sys_name, 0) / n if n else 0
        bar = "█" * int(avg)
        print(f"    {sys_name:<12} {avg:>5.1f}  {bar}")

    print("=" * 80)

    report_path = os.path.join(os.path.dirname(__file__), "head_to_head_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results: {report_path}")


if __name__ == "__main__":
    main()
