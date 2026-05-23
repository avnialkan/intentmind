"""
Intentmind Cognitive Benchmark Suite
====================================
Tests Intentmind (GraphRAG) vs Baseline (Vector RAG) across 5 cognitive dimensions:
1. Single-Hop (Factual Recall) - Both should tie
2. Multi-Hop (Chain Reasoning) - Intentmind should dominate
3. Associative (Cross-concept Synthesis) - Intentmind should dominate
4. Temporal/Session (Memory across turns) - Intentmind should dominate
5. Counterfactual (Causal Reasoning) - Intentmind should dominate
"""
import os
import sys
import json
import time
from typing import Dict, List
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
            {"role": "system", "content": "You are a literary analysis assistant. Answer based ONLY on the provided context. If the context doesn't contain the answer, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content


def evaluate_answer(query: str, answer: str, context: str, expected_contains: list) -> Dict:
    """Evaluate with both LLM judgment AND keyword ground-truth checking."""
    # 1. Keyword ground truth check
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected_contains if kw.lower() in answer_lower)
    keyword_score = round(hits / len(expected_contains) * 5, 1) if expected_contains else 0

    # 2. LLM-as-Judge (Granular RAGAS)
    prompt = f"""You are an expert evaluator. Score the Answer on these metrics (1-5 each):

1. **Faithfulness**: Is the answer grounded in the context? (5=fully grounded, 1=hallucinated)
2. **Answer Relevance**: Does it directly answer the question? (5=perfect, 1=off-topic)  
3. **Completeness**: Does the answer cover ALL aspects of the question? (5=comprehensive, 1=partial)
4. **Reasoning Depth**: Does the answer show understanding of causal chains and connections? (5=deep insight, 1=surface-level)

Context:
{context[:3000]}

Question: {query}

Answer: {answer}

Respond ONLY as JSON:
{{"faithfulness": 5, "relevance": 5, "completeness": 4, "reasoning_depth": 3, "explanation": "..."}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        scores = json.loads(response.choices[0].message.content)
        scores["keyword_recall"] = keyword_score
        return scores
    except Exception as e:
        return {"faithfulness": 0, "relevance": 0, "completeness": 0, "reasoning_depth": 0, "keyword_recall": keyword_score, "explanation": str(e)}


def run_single_query(memory: IntentmindMemory, query: str, expected: list) -> Dict:
    """Run a single query through both Intentmind and Baseline, return scores."""
    # Intentmind (GraphRAG)
    t0 = time.perf_counter()
    im_result = memory.query(query)
    im_latency = time.perf_counter() - t0
    im_context = im_result["prompt"]
    im_answer = generate_answer(query, im_context)
    im_scores = evaluate_answer(query, im_answer, im_context, expected)
    im_scores["latency"] = round(im_latency, 3)
    im_scores["context_chars"] = len(im_context)
    im_scores["memory_items"] = len(im_result["memories"]["items"])

    # Baseline RAG (Pure Vector)
    t0 = time.perf_counter()
    query_emb = memory.embedder.embed(query)
    raw_chunks = [c for _, c in memory._store.search_chunks(query_emb, top_k=5)]
    base_latency = time.perf_counter() - t0
    base_context = "\n\n".join([c.text for c in raw_chunks])
    base_answer = generate_answer(query, base_context)
    base_scores = evaluate_answer(query, base_answer, base_context, expected)
    base_scores["latency"] = round(base_latency, 3)
    base_scores["context_chars"] = len(base_context)
    base_scores["memory_items"] = len(raw_chunks)

    return {
        "intentmind": {"answer": im_answer, "scores": im_scores},
        "baseline": {"answer": base_answer, "scores": base_scores}
    }


def run_session_query(memory: IntentmindMemory, queries: list, expected: list) -> Dict:
    """Run a multi-turn session. Intentmind accumulates energy; Baseline is stateless."""
    # Intentmind: run all queries sequentially (energy accumulates)
    im_answers = []
    for q in queries:
        im_result = memory.query(q)
        im_context = im_result["prompt"]
        ans = generate_answer(q, im_context)
        im_answers.append(ans)

    # For scoring, evaluate the FINAL answer (which benefits from prior activations)
    final_im_scores = evaluate_answer(queries[-1], im_answers[-1], im_context, expected)
    final_im_scores["context_chars"] = len(im_context)
    final_im_scores["memory_items"] = len(im_result["memories"]["items"])

    # Baseline: each query is independent (no memory between turns)
    query_emb = memory.embedder.embed(queries[-1])
    raw_chunks = [c for _, c in memory._store.search_chunks(query_emb, top_k=5)]
    base_context = "\n\n".join([c.text for c in raw_chunks])
    base_answer = generate_answer(queries[-1], base_context)
    final_base_scores = evaluate_answer(queries[-1], base_answer, base_context, expected)
    final_base_scores["context_chars"] = len(base_context)
    final_base_scores["memory_items"] = len(raw_chunks)

    return {
        "intentmind": {"answer": im_answers[-1], "scores": final_im_scores, "session_answers": im_answers},
        "baseline": {"answer": base_answer, "scores": final_base_scores}
    }


def main():
    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "scale_test.json")
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "cognitive_benchmark.json")

    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        return

    print("Loading Intentmind Memory Engine...")
    memory = IntentmindMemory.load(db_path)
    stats = memory._store.stats()
    print(f"Graph: {stats['intents']} Nodes | {stats['edges']} Edges | {stats['chunks']} Chunks\n")

    with open(dataset_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    results_by_category = {}
    all_results = []

    for i, test in enumerate(benchmark):
        category = test["category"]
        if category not in results_by_category:
            results_by_category[category] = {"intentmind_total": 0, "baseline_total": 0, "count": 0}

        if category == "temporal_session":
            query_text = " -> ".join(test["query_sequence"])
            expected = test.get("expected_final_contains", [])
            print(f"[{i+1}/{len(benchmark)}] ({category}) Session: {query_text[:80]}...")
            result = run_session_query(memory, test["query_sequence"], expected)
        else:
            query_text = test["query"]
            expected = test.get("expected_answer_contains", [])
            print(f"[{i+1}/{len(benchmark)}] ({category}) {query_text[:80]}...")
            result = run_single_query(memory, query_text, expected)

        # Compute composite score (weighted average)
        for system in ["intentmind", "baseline"]:
            s = result[system]["scores"]
            composite = (
                s.get("faithfulness", 0) * 0.25 +
                s.get("relevance", 0) * 0.20 +
                s.get("completeness", 0) * 0.25 +
                s.get("reasoning_depth", 0) * 0.20 +
                s.get("keyword_recall", 0) * 0.10
            )
            result[system]["scores"]["composite"] = round(composite, 2)

        im_comp = result["intentmind"]["scores"]["composite"]
        base_comp = result["baseline"]["scores"]["composite"]
        winner = "Intentmind" if im_comp > base_comp else ("Baseline" if base_comp > im_comp else "Tie")

        results_by_category[category]["intentmind_total"] += im_comp
        results_by_category[category]["baseline_total"] += base_comp
        results_by_category[category]["count"] += 1

        print(f"  Intentmind: {im_comp:.2f} | Baseline: {base_comp:.2f} | Winner: {winner}")

        all_results.append({
            "category": category,
            "query": query_text,
            "winner": winner,
            **result
        })

    # Final Report
    print("\n" + "=" * 70)
    print("        INTENTMIND COGNITIVE BENCHMARK REPORT")
    print("=" * 70)
    print(f"{'Category':<22} {'Intentmind':>12} {'Baseline':>12} {'Winner':>12}")
    print("-" * 58)

    total_im = 0
    total_base = 0
    total_count = 0

    for cat, data in results_by_category.items():
        n = data["count"]
        im_avg = data["intentmind_total"] / n if n else 0
        base_avg = data["baseline_total"] / n if n else 0
        w = "Intentmind" if im_avg > base_avg else ("Baseline" if base_avg > im_avg else "Tie")
        print(f"{cat:<22} {im_avg:>10.2f}/5 {base_avg:>10.2f}/5 {w:>12}")
        total_im += data["intentmind_total"]
        total_base += data["baseline_total"]
        total_count += n

    print("-" * 58)
    overall_im = total_im / total_count if total_count else 0
    overall_base = total_base / total_count if total_count else 0
    overall_winner = "INTENTMIND" if overall_im > overall_base else ("BASELINE" if overall_base > overall_im else "TIE")
    print(f"{'OVERALL':<22} {overall_im:>10.2f}/5 {overall_base:>10.2f}/5 {overall_winner:>12}")
    print("=" * 70)

    # Save full results
    report_path = os.path.join(os.path.dirname(__file__), "cognitive_benchmark_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to: {report_path}")


if __name__ == "__main__":
    main()
