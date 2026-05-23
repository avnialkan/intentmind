"""
Cross-Domain Confusion Benchmark
=================================
The ULTIMATE test: Both Pride & Prejudice AND Frankenstein are in the same memory.
Tests whether the system can:
1. Distinguish between sources (two different "Elizabeth" characters!)
2. Resist noise from the wrong domain
3. Synthesize across domains when asked
4. Maintain deep recall within a single domain despite noise
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

FRANKENSTEIN_KEYWORDS = ["victor", "creature", "monster", "frankenstein", "clerval", "justine", "william frankenstein", "laboratory", "galvanism", "arctic"]
PRIDE_KEYWORDS = ["bennet", "darcy", "bingley", "pemberley", "longbourn", "wickham", "lydia", "jane austen", "netherfield", "meryton"]


def generate_answer(query: str, context: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a literary analysis assistant. Answer based ONLY on the provided context. Be specific about which novel and characters you are discussing."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.0
    )
    return response.choices[0].message.content


def check_source_contamination(answer: str, correct_source: str) -> Dict:
    """Check if the answer incorrectly mixes sources."""
    answer_lower = answer.lower()
    pride_hits = sum(1 for kw in PRIDE_KEYWORDS if kw in answer_lower)
    frank_hits = sum(1 for kw in FRANKENSTEIN_KEYWORDS if kw in answer_lower)
    
    if correct_source == "pride":
        contamination = frank_hits / max(1, frank_hits + pride_hits) * 100
        correct_domain_score = pride_hits
    elif correct_source == "frank":
        contamination = pride_hits / max(1, pride_hits + frank_hits) * 100
        correct_domain_score = frank_hits
    else:  # "both" - both should be present
        contamination = 0
        correct_domain_score = pride_hits + frank_hits
    
    return {
        "pride_mentions": pride_hits,
        "frank_mentions": frank_hits,
        "contamination_pct": round(contamination, 1),
        "correct_domain_hits": correct_domain_score
    }


def evaluate_answer(query: str, answer: str, context: str, expected: list) -> Dict:
    """Full evaluation with LLM judge."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in expected if kw.lower() in answer_lower)
    keyword_score = round(hits / len(expected) * 5, 1) if expected else 0

    prompt = f"""You are an expert evaluator of RAG systems that contain MULTIPLE knowledge sources.

Score the Answer on these metrics (1-5 each):
1. **Faithfulness**: Is the answer grounded in the context provided? (5=fully, 1=hallucinated)
2. **Relevance**: Does it directly answer the question? (5=perfect, 1=off-topic)
3. **Source Accuracy**: Does the answer correctly attribute information to the right novel/source without mixing up characters or events from different books? (5=perfectly attributed, 1=sources completely confused)
4. **Completeness**: Does it cover all aspects of the question? (5=comprehensive, 1=partial)

Context (first 3000 chars):
{context[:3000]}

Question: {query}
Answer: {answer}

Respond ONLY as JSON:
{{"faithfulness": 5, "relevance": 5, "source_accuracy": 4, "completeness": 4, "explanation": "..."}}"""

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
        return {"faithfulness": 0, "relevance": 0, "source_accuracy": 0, "completeness": 0, "keyword_recall": keyword_score, "explanation": str(e)}


def main():
    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "mixed_domain_test.json")
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "cross_domain_benchmark.json")

    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found. Run mixed_domain_ingest.py first.")
        return

    print("Loading Mixed-Domain Memory (P&P + Frankenstein)...")
    memory = IntentmindMemory.load(db_path)
    stats = memory._store.stats()
    print(f"Graph: {stats['intents']} Nodes | {stats['edges']} Edges | {stats['chunks']} Chunks\n")

    with open(dataset_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    results_by_category = {}
    all_results = []

    for i, test in enumerate(benchmark):
        category = test["category"]
        query = test["query"]
        correct_source = test.get("correct_source", "both")
        expected = test.get("expected_contains", test.get("expected_pride", []) + test.get("expected_frank", []))

        if category not in results_by_category:
            results_by_category[category] = {"im_composite": 0, "base_composite": 0, "im_contam": 0, "base_contam": 0, "count": 0}

        print(f"[{i+1}/{len(benchmark)}] ({category}) {query[:80]}...")

        # === INTENTMIND ===
        t0 = time.perf_counter()
        im_result = memory.query(query)
        im_latency = time.perf_counter() - t0
        im_context = im_result["prompt"]
        im_answer = generate_answer(query, im_context)
        im_scores = evaluate_answer(query, im_answer, im_context, expected)
        im_contam = check_source_contamination(im_answer, correct_source)
        im_scores["contamination"] = im_contam
        im_scores["latency"] = round(im_latency, 3)

        # === BASELINE RAG ===
        t0 = time.perf_counter()
        query_emb = memory.embedder.embed(query)
        raw_chunks = [c for _, c in memory._store.search_chunks(query_emb, top_k=5)]
        base_latency = time.perf_counter() - t0
        base_context = "\n\n".join([c.text for c in raw_chunks])
        base_answer = generate_answer(query, base_context)
        base_scores = evaluate_answer(query, base_answer, base_context, expected)
        base_contam = check_source_contamination(base_answer, correct_source)
        base_scores["contamination"] = base_contam
        base_scores["latency"] = round(base_latency, 3)

        # Composite (source_accuracy is now 30% weight - the key differentiator)
        for system_name, scores in [("intentmind", im_scores), ("baseline", base_scores)]:
            composite = (
                scores.get("faithfulness", 0) * 0.20 +
                scores.get("relevance", 0) * 0.15 +
                scores.get("source_accuracy", 0) * 0.30 +
                scores.get("completeness", 0) * 0.20 +
                scores.get("keyword_recall", 0) * 0.15
            )
            scores["composite"] = round(composite, 2)

        im_comp = im_scores["composite"]
        base_comp = base_scores["composite"]
        winner = "Intentmind" if im_comp > base_comp else ("Baseline" if base_comp > im_comp else "Tie")

        results_by_category[category]["im_composite"] += im_comp
        results_by_category[category]["base_composite"] += base_comp
        results_by_category[category]["im_contam"] += im_contam["contamination_pct"]
        results_by_category[category]["base_contam"] += base_contam["contamination_pct"]
        results_by_category[category]["count"] += 1

        print(f"  IM: {im_comp:.2f} (contam: {im_contam['contamination_pct']}%) | BASE: {base_comp:.2f} (contam: {base_contam['contamination_pct']}%) | {winner}")

        all_results.append({
            "category": category, "query": query, "correct_source": correct_source, "winner": winner,
            "intentmind": {"answer": im_answer, "scores": im_scores},
            "baseline": {"answer": base_answer, "scores": base_scores}
        })

    # === FINAL REPORT ===
    print("\n" + "=" * 80)
    print("     CROSS-DOMAIN CONFUSION BENCHMARK REPORT")
    print("     Pride & Prejudice + Frankenstein in ONE Memory")
    print("=" * 80)
    print(f"{'Category':<26} {'Intentmind':>12} {'Baseline':>12} {'IM Contam%':>12} {'Base Contam%':>12} {'Winner':>10}")
    print("-" * 84)

    total_im = total_base = total_im_c = total_base_c = total_n = 0

    for cat, d in results_by_category.items():
        n = d["count"]
        im_avg = d["im_composite"] / n
        base_avg = d["base_composite"] / n
        im_c = d["im_contam"] / n
        base_c = d["base_contam"] / n
        w = "Intentmind" if im_avg > base_avg else ("Baseline" if base_avg > im_avg else "Tie")
        print(f"{cat:<26} {im_avg:>10.2f}/5 {base_avg:>10.2f}/5 {im_c:>10.1f}% {base_c:>10.1f}% {w:>10}")
        total_im += d["im_composite"]
        total_base += d["base_composite"]
        total_im_c += d["im_contam"]
        total_base_c += d["base_contam"]
        total_n += n

    print("-" * 84)
    oi = total_im / total_n
    ob = total_base / total_n
    ow = "INTENTMIND" if oi > ob else ("BASELINE" if ob > oi else "TIE")
    print(f"{'OVERALL':<26} {oi:>10.2f}/5 {ob:>10.2f}/5 {total_im_c/total_n:>10.1f}% {total_base_c/total_n:>10.1f}% {ow:>10}")
    print("=" * 80)

    report_path = os.path.join(os.path.dirname(__file__), "cross_domain_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results: {report_path}")


if __name__ == "__main__":
    main()
