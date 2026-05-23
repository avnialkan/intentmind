import os
import sys

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.intentmind.runtime import IntentmindMemory
from src.intentmind.benchmark.runner import BenchmarkRunner
from src.intentmind.benchmark.utils import load_fixture, build_fixture_extractor, load_fixture_and_map

def main():
    fixture_path = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "real_world_benchmark_v1.json")
    
    # 1. Load raw data just to build the extractor
    raw_data = load_fixture(fixture_path)
    extractor = build_fixture_extractor(raw_data)
    
    # 2. Init memory with fixture extractor and FakeEmbedder
    mem = IntentmindMemory(is_test=True, extractor=extractor)
    
    # 3. Ingest and map
    print(f"Ingesting memories from {fixture_path}...")
    mapped_data = load_fixture_and_map(mem, fixture_path)
    
    # 4. Run Benchmark
    print("Running queries...")
    runner = BenchmarkRunner(mem)
    results = runner.run_suite(mapped_data.get("queries", []))
    
    # 5. Generate report
    report_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_report.md")
    report = runner.generate_markdown_report(results)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        
    print(f"Benchmark finished! Report written to: {report_path}")
    print(f"Scores -> Classic RAG F1: {results['summary']['classic_rag_avg_f1']} | Intentmind F1: {results['summary']['intentmind_avg_f1']}")
    print(f"Intentmind token savings: {results['summary']['intentmind_avg_token_saving_pct']}%")

if __name__ == "__main__":
    main()
