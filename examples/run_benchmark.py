import json
import os
from intentmind import IntentmindMemory
from intentmind.benchmark import BenchmarkRunner

def main():
    print("Loading benchmark fixture...")
    fixture_path = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "benchmark_v1.json")
    
    print("Ingesting memories and resolving deduplication mappings...")
    from intentmind.benchmark.utils import build_fixture_extractor, load_fixture, load_fixture_and_map
    fixture_data = load_fixture(fixture_path)
    extractor = build_fixture_extractor(fixture_data)

    print("Initializing IntentmindMemory (Test Mode + fixture LLM extractor)...")
    mem = IntentmindMemory(is_test=True, extractor=extractor)
    data = load_fixture_and_map(mem, fixture_path)

    # Simulate some time passage and consolidation to test meaning shift/decay
    # mem.tick(hours=24.0)

    print("Starting Benchmark Runner...")
    runner = BenchmarkRunner(mem)
    results = runner.run_suite(data["queries"])

    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    summary = results["summary"]
    
    print(f"Total Queries: {summary['total_queries']}")
    print("-" * 50)
    print("CLASSIC RAG")
    print(f"  Precision : {summary['classic_rag_avg_precision']}")
    print(f"  Recall    : {summary['classic_rag_avg_recall']}")
    print(f"  Latency   : {summary['classic_rag_avg_latency_ms']} ms")
    print("-" * 50)
    print("INTENTMIND")
    print(f"  Precision : {summary['intentmind_avg_precision']}")
    print(f"  Recall    : {summary['intentmind_avg_recall']}")
    print(f"  Latency   : {summary['intentmind_avg_latency_ms']} ms")
    print(f"  Token Save: {summary['intentmind_avg_token_saving_pct']}%")
    print("="*50)

    out_path = "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {out_path}")

if __name__ == "__main__":
    main()
