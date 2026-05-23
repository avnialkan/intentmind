import argparse
import json
import os
from pathlib import Path

from intentmind import IntentmindMemory
from intentmind.benchmark import BenchmarkRunner
from intentmind.benchmark.utils import build_fixture_extractor, load_fixture, load_fixture_and_map


def load_dotenv_simple(path: str = ".env"):
    """Load .env file into os.environ without external dependencies."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def build_embedder(use_fake: bool):
    if use_fake:
        from intentmind.embeddings import FakeEmbedder

        return FakeEmbedder(), True

    try:
        from intentmind.embeddings import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(), False
    except Exception as exc:
        print(f"[warning] Could not load SentenceTransformerEmbedder: {exc}")
        print("[warning] Falling back to FakeEmbedder. Results are useful only for regression, not public claims.")
        from intentmind.embeddings import FakeEmbedder

        return FakeEmbedder(), True


def main():
    parser = argparse.ArgumentParser(description="Run the real-world Intentmind vs classic RAG benchmark.")
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--fixture",
        default=str(root / "tests" / "fixtures" / "real_world_benchmark_v1.json"),
        help="Benchmark fixture JSON path.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Classic RAG top-k.")
    parser.add_argument("--index-type", choices=["exact", "faiss"], default="exact", help="Intent index backend.")
    parser.add_argument("--fake", action="store_true", help="Use FakeEmbedder for fast deterministic smoke runs.")
    parser.add_argument("--llm", action="store_true", help="Use real LLM extractor (OpenAI API) instead of fixture extractor.")
    parser.add_argument("--model", default=None, help="OpenAI model name for LLM extractor. Defaults to OPENAI_MODEL env var or gpt-4o-mini.")
    parser.add_argument("--json-out", default=str(root / "real_world_benchmark_results.json"))
    parser.add_argument("--report-out", default=str(root / "REAL_WORLD_BENCHMARK_REPORT.md"))
    args = parser.parse_args()

    # Load .env for API keys
    load_dotenv_simple(str(root / ".env"))

    fixture_path = Path(args.fixture)
    fixture_data = load_fixture(str(fixture_path))
    embedder, used_fake = build_embedder(args.fake)

    # Determine extractor mode
    if args.llm:
        extractor = None  # Let IntentmindMemory use its built-in LLM extractor
        model = args.model or os.environ.get("OPENAI_MODEL")
        core_extractor = "llm"
        extractor_label = f"LLM ({model})"

        if not os.environ.get("OPENAI_API_KEY"):
            print("[ERROR] --llm requires OPENAI_API_KEY in environment or .env file.")
            return
    else:
        extractor = build_fixture_extractor(fixture_data)
        model = None
        core_extractor = "llm"  # Will be overridden to deterministic by fixture extractor
        extractor_label = "fixture (deterministic)"

    print("Loading real-world benchmark fixture...")
    print(f"Memories : {len(fixture_data.get('memories', []))}")
    print(f"Queries  : {len(fixture_data.get('queries', []))}")
    print(f"Embedder : {getattr(embedder, 'name', embedder.__class__.__name__)}")
    print(f"Index    : {args.index_type}")
    print(f"Extractor: {extractor_label}")

    mem = IntentmindMemory(
        embedder=embedder,
        is_test=used_fake,
        extractor=extractor,
        index_type=args.index_type,
        core_extractor=core_extractor,
        model=model,
    )
    data = load_fixture_and_map(mem, str(fixture_path))

    runner = BenchmarkRunner(mem)
    results = runner.run_suite(data["queries"], top_k=args.top_k)

    # Tag the results with extractor info
    results["summary"]["extractor"] = extractor_label
    if args.llm:
        results["summary"]["llm_model"] = model

    report = runner.generate_markdown_report(results, title="Intentmind Real World Benchmark Report")

    json_path = Path(args.json_out)
    report_path = Path(args.report_out)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    summary = results["summary"]
    print("\n" + "=" * 72)
    print("REAL WORLD BENCHMARK RESULTS")
    print("=" * 72)
    print(f"Total Queries: {summary['total_queries']} | Top-K: {summary['top_k']}")
    print(f"Extractor: {extractor_label}")
    print("-" * 72)
    print("CLASSIC RAG")
    print(f"  Precision : {summary['classic_rag_avg_precision']}")
    print(f"  Recall    : {summary['classic_rag_avg_recall']}")
    print(f"  F1        : {summary['classic_rag_avg_f1']}")
    print(f"  Hit@K     : {summary['classic_rag_avg_hit_at_k']}")
    print(f"  MRR       : {summary['classic_rag_avg_mrr']}")
    print(f"  Tokens    : {summary['classic_rag_avg_tokens']}")
    print(f"  p95 Lat   : {summary['classic_rag_p95_latency_ms']} ms")
    print("-" * 72)
    print("INTENTMIND")
    print(f"  Precision : {summary['intentmind_avg_precision']}")
    print(f"  Recall    : {summary['intentmind_avg_recall']}")
    print(f"  F1        : {summary['intentmind_avg_f1']}")
    print(f"  Hit@K     : {summary['intentmind_avg_hit_at_k']}")
    print(f"  MRR       : {summary['intentmind_avg_mrr']}")
    print(f"  Tokens    : {summary['intentmind_avg_tokens']}")
    print(f"  Token Save: {summary['intentmind_avg_token_saving_pct']}%")
    print(f"  p95 Lat   : {summary['intentmind_p95_latency_ms']} ms")

    # Print latency breakdown
    lb = summary.get("intentmind_latency_breakdown", {})
    if lb:
        print("-" * 72)
        print("LATENCY BREAKDOWN (avg)")
        for key, val in lb.items():
            print(f"  {key:20s}: {val:>8.2f} ms")

    print("=" * 72)
    print(f"Detailed JSON: {json_path}")
    print(f"Markdown report: {report_path}")
    if used_fake:
        print("\n[warning] FakeEmbedder was used. Do not treat this run as a public benchmark.")
    if args.llm:
        print(f"\n[info] LLM extractor used: {model}")
        print("[info] Results depend on LLM extraction quality and are NOT deterministic across runs.")


if __name__ == "__main__":
    main()
