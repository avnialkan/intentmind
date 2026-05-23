"""
Microsoft GraphRAG-Style Benchmark
====================================
Replicates Microsoft's exact evaluation methodology from their GraphRAG paper:
- Global Sensemaking Questions (the type GraphRAG was designed for)
- LLM-as-a-Judge scoring on: Comprehensiveness, Diversity, Empowerment
- Head-to-head comparison: Intentmind vs Baseline RAG

Reference: https://microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/
"""
import os
import sys
import json
import time
from typing import Dict
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from intentmind import IntentmindMemory
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
client = OpenAI()
MODEL_NAME = os.getenv("OPENAI_MODEL")


def generate_answer(query: str, context: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful assistant performing a comprehensive analysis. Use ONLY the provided context to craft a detailed, multi-faceted answer. Cover as many aspects and perspectives as possible."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.0,
        max_completion_tokens=1500
    )
    return response.choices[0].message.content


def graphrag_evaluate(query: str, answer_a: str, answer_b: str) -> Dict:
    """
    Microsoft GraphRAG's exact evaluation method: 
    LLM-as-a-judge comparing two answers on Comprehensiveness, Diversity, Empowerment.
    System A = Intentmind, System B = Baseline RAG.
    """
    prompt = f"""You are an expert evaluator comparing two answers to the same question.
Evaluate which answer is better on these three criteria (as used by Microsoft Research's GraphRAG evaluation):

1. **Comprehensiveness**: Which answer covers more relevant aspects, details, and facets of the question? Does it address the full scope?
2. **Diversity**: Which answer provides more varied perspectives, viewpoints, or angles on the topic? Does it explore different dimensions?
3. **Empowerment**: Which answer better helps the reader understand the topic and make informed judgments? Is it more insightful and useful?

For EACH criterion, you must choose: "System A", "System B", or "Tie".
Also provide an overall winner.

Question: {query}

=== System A Answer ===
{answer_a}

=== System B Answer ===
{answer_b}

Respond ONLY as JSON:
{{
    "comprehensiveness": "System A",
    "diversity": "System B", 
    "empowerment": "System A",
    "overall_winner": "System A",
    "reasoning": "Detailed explanation of why..."
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
        return {"comprehensiveness": "Error", "diversity": "Error", "empowerment": "Error", "overall_winner": "Error", "reasoning": str(e)}


def main():
    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "mixed_domain_test.json")
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "graphrag_style_benchmark.json")

    print("=" * 70)
    print("  MICROSOFT GRAPHRAG-STYLE BENCHMARK")
    print("  Metrics: Comprehensiveness | Diversity | Empowerment")
    print("=" * 70)

    print("\nLoading Mixed-Domain Memory (P&P + Frankenstein)...")
    memory = IntentmindMemory.load(db_path)
    stats = memory._store.stats()
    print(f"Graph: {stats['intents']} Nodes | {stats['edges']} Edges | {stats['chunks']} Chunks\n")

    with open(dataset_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    results = []
    wins = {"System A": 0, "System B": 0, "Tie": 0}
    metric_wins = {
        "comprehensiveness": {"System A": 0, "System B": 0, "Tie": 0},
        "diversity": {"System A": 0, "System B": 0, "Tie": 0},
        "empowerment": {"System A": 0, "System B": 0, "Tie": 0},
    }

    for i, test in enumerate(benchmark):
        query = test["query"]
        qtype = test["type"]
        print(f"[{i+1}/{len(benchmark)}] ({qtype}) {query[:70]}...")

        # System A: Intentmind (GraphRAG)
        t0 = time.perf_counter()
        im_result = memory.query(query)
        im_latency = time.perf_counter() - t0
        im_context = im_result["prompt"]
        im_answer = generate_answer(query, im_context)

        # System B: Baseline RAG (Vector Only)
        t0 = time.perf_counter()
        query_emb = memory.embedder.embed(query)
        raw_chunks = [c for _, c in memory._store.search_chunks(query_emb, top_k=5)]
        base_latency = time.perf_counter() - t0
        base_context = "\n\n".join([c.text for c in raw_chunks])
        base_answer = generate_answer(query, base_context)

        # Evaluate (GraphRAG style)
        evaluation = graphrag_evaluate(query, im_answer, base_answer)

        overall = evaluation.get("overall_winner", "Tie")
        wins[overall] = wins.get(overall, 0) + 1

        for metric in ["comprehensiveness", "diversity", "empowerment"]:
            w = evaluation.get(metric, "Tie")
            metric_wins[metric][w] = metric_wins[metric].get(w, 0) + 1

        winner_label = "INTENTMIND" if overall == "System A" else ("BASELINE" if overall == "System B" else "TIE")
        print(f"  Winner: {winner_label}")
        print(f"  Comp: {evaluation.get('comprehensiveness', '?')} | Div: {evaluation.get('diversity', '?')} | Emp: {evaluation.get('empowerment', '?')}")

        results.append({
            "id": test["id"],
            "type": qtype,
            "query": query,
            "intentmind_answer": im_answer,
            "baseline_answer": base_answer,
            "intentmind_latency": round(im_latency, 3),
            "baseline_latency": round(base_latency, 3),
            "intentmind_context_chars": len(im_context),
            "baseline_context_chars": len(base_context),
            "intentmind_memory_items": len(im_result["memories"]["items"]),
            "evaluation": evaluation,
        })

    # Final Report
    print("\n" + "=" * 70)
    print("  GRAPHRAG-STYLE BENCHMARK FINAL REPORT")
    print("  (System A = Intentmind | System B = Baseline RAG)")
    print("=" * 70)

    print(f"\n  OVERALL WINS:")
    print(f"    Intentmind (System A): {wins.get('System A', 0)}/{len(benchmark)}")
    print(f"    Baseline   (System B): {wins.get('System B', 0)}/{len(benchmark)}")
    print(f"    Tie:                   {wins.get('Tie', 0)}/{len(benchmark)}")

    print(f"\n  PER-METRIC BREAKDOWN:")
    for metric in ["comprehensiveness", "diversity", "empowerment"]:
        mw = metric_wins[metric]
        print(f"    {metric.upper():<20} IM: {mw.get('System A', 0)} | BASE: {mw.get('System B', 0)} | Tie: {mw.get('Tie', 0)}")

    # By question type
    print(f"\n  BY QUESTION TYPE:")
    type_wins = {}
    for r in results:
        t = r["type"]
        if t not in type_wins:
            type_wins[t] = {"System A": 0, "System B": 0, "Tie": 0, "total": 0}
        w = r["evaluation"].get("overall_winner", "Tie")
        type_wins[t][w] = type_wins[t].get(w, 0) + 1
        type_wins[t]["total"] += 1

    for t, tw in type_wins.items():
        print(f"    {t:<25} IM: {tw.get('System A', 0)}/{tw['total']} | BASE: {tw.get('System B', 0)}/{tw['total']} | Tie: {tw.get('Tie', 0)}/{tw['total']}")

    print("=" * 70)

    report_path = os.path.join(os.path.dirname(__file__), "graphrag_style_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results: {report_path}")


if __name__ == "__main__":
    main()
