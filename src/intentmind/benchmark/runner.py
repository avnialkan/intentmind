import time
from math import ceil
from typing import List, Dict, Any
from ..runtime import IntentmindMemory
from ..embeddings import cosine_similarity

class BenchmarkRunner:
    def __init__(self, memory: IntentmindMemory):
        self.memory = memory

    def classic_rag_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Simulates a classic vector search RAG without Intentmind features."""
        query_emb = self.memory.embedder.embed(query)
        results = []
        for chunk in self.memory._store.chunks.values():
            sim = cosine_similarity(query_emb, chunk.embedding)
            results.append({"chunk_id": chunk.chunk_id, "score": sim, "text": chunk.text})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def evaluate_query(
        self,
        query: str,
        expected_chunks: List[str],
        expected_intents: List[str] = None,
        expected_reason: str | None = None,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Evaluates a single query against both Classic RAG and Intentmind."""
        
        # 1. Classic RAG
        start_time = time.perf_counter()
        rag_results = self.classic_rag_search(query, top_k=top_k)
        rag_latency = time.perf_counter() - start_time
        
        rag_retrieved_ids = [r["chunk_id"] for r in rag_results]
        rag_tokens = sum(len(r["text"].split()) for r in rag_results)

        # 2. Intentmind
        start_time = time.perf_counter()
        im_results = self.memory.query(query)
        im_latency = time.perf_counter() - start_time
        
        im_items = im_results["memories"]["items"]
        im_retrieved_ids = [item["chunk_id"] for item in im_items]
        im_tokens = sum(len(item["text"].split()) for item in im_items)
        
        # Calculate Metrics
        expected_set = set(expected_chunks)
        
        rag_precision, rag_recall = self._precision_recall(rag_retrieved_ids, expected_set)
        im_precision, im_recall = self._precision_recall(im_retrieved_ids, expected_set)
        rag_f1 = self._f1(rag_precision, rag_recall)
        im_f1 = self._f1(im_precision, im_recall)
        
        # Token Saving (lower is better, but saving % is higher = better)
        token_saving = 0.0
        if rag_tokens > 0:
            token_saving = (rag_tokens - im_tokens) / rag_tokens

        return {
            "query": query,
            "expected_chunks": expected_chunks,
            "classic_rag": {
                "latency_ms": round(rag_latency * 1000, 2),
                "precision": round(rag_precision, 3),
                "recall": round(rag_recall, 3),
                "f1": round(rag_f1, 3),
                "hit_at_k": self._hit_at_k(rag_retrieved_ids, expected_set, top_k),
                "mrr": round(self._mrr(rag_retrieved_ids, expected_set), 3),
                "retrieved_count": len(rag_retrieved_ids),
                "tokens": rag_tokens,
                "false_positives": [cid for cid in rag_retrieved_ids if cid not in expected_set],
                "false_negatives": [cid for cid in expected_chunks if cid not in set(rag_retrieved_ids)],
                "retrieved": [
                    {
                        "rank": idx + 1,
                        "chunk_id": item["chunk_id"],
                        "score": round(item["score"], 4),
                        "text": item["text"],
                    }
                    for idx, item in enumerate(rag_results)
                ],
            },
            "intentmind": {
                "latency_ms": round(im_latency * 1000, 2),
                "latency_breakdown": im_results.get("latency_breakdown", {}),
                "precision": round(im_precision, 3),
                "recall": round(im_recall, 3),
                "f1": round(im_f1, 3),
                "hit_at_k": self._hit_at_k(im_retrieved_ids, expected_set, top_k),
                "mrr": round(self._mrr(im_retrieved_ids, expected_set), 3),
                "retrieved_count": len(im_retrieved_ids),
                "tokens": im_tokens,
                "token_saving_pct": round(token_saving * 100, 2),
                "rejected_count": im_results["memories"]["rejected"],
                "false_positives": [cid for cid in im_retrieved_ids if cid not in expected_set],
                "false_negatives": [cid for cid in expected_chunks if cid not in set(im_retrieved_ids)],
                "direct_count": im_results["memories"]["direct"],
                "associated_count": im_results["memories"]["associated"],
                "weak_echo_count": im_results["memories"]["weak_echo"],
                "retrieved": [
                    {
                        "rank": idx + 1,
                        "chunk_id": item["chunk_id"],
                        "score": round(item["score"], 4),
                        "layer": item["layer"],
                        "intent": item["intent"],
                        "called_by": item.get("called_by"),
                        "reason": item.get("reason"),
                        "path": item.get("path", []),
                        "edge": item.get("edge", {}),
                        "text": item["text"],
                    }
                    for idx, item in enumerate(im_items)
                ],
                "trace": im_results.get("trace", []),
            },
            "analysis": {
                "expected_intents": expected_intents or [],
                "expected_reason": expected_reason or "",
                "intentmind_paths": [
                    item.get("path", [])
                    for item in im_items
                    if item.get("path")
                ],
            },
        }

    def run_suite(self, queries: List[Dict], top_k: int = 5) -> Dict[str, Any]:
        """Runs the benchmark on a list of test cases."""
        results = []
        for q_data in queries:
            res = self.evaluate_query(
                query=q_data["query"],
                expected_chunks=q_data["expected_chunks"],
                expected_intents=q_data.get("expected_intents", []),
                expected_reason=q_data.get("expected_reason"),
                top_k=top_k,
            )
            results.append(res)
            
        # Aggregate
        def avg(key1, key2):
            return sum(r[key1][key2] for r in results) / len(results)

        # Aggregate latency breakdown
        breakdown_keys = ["embed_query_ms", "emotion_ms", "extractor_ms", "recall_ms", "prompt_ms"]
        def breakdown_avg(key):
            vals = [r["intentmind"].get("latency_breakdown", {}).get(key, 0.0) for r in results]
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        summary = {
            "total_queries": len(queries),
            "top_k": top_k,
            "classic_rag_avg_precision": round(avg("classic_rag", "precision"), 3),
            "classic_rag_avg_recall": round(avg("classic_rag", "recall"), 3),
            "classic_rag_avg_f1": round(avg("classic_rag", "f1"), 3),
            "classic_rag_avg_hit_at_k": round(avg("classic_rag", "hit_at_k"), 3),
            "classic_rag_avg_mrr": round(avg("classic_rag", "mrr"), 3),
            "classic_rag_avg_latency_ms": round(avg("classic_rag", "latency_ms"), 3),
            "classic_rag_p50_latency_ms": self._percentile([r["classic_rag"]["latency_ms"] for r in results], 50),
            "classic_rag_p95_latency_ms": self._percentile([r["classic_rag"]["latency_ms"] for r in results], 95),
            "classic_rag_avg_tokens": round(avg("classic_rag", "tokens"), 3),
            "intentmind_avg_precision": round(avg("intentmind", "precision"), 3),
            "intentmind_avg_recall": round(avg("intentmind", "recall"), 3),
            "intentmind_avg_f1": round(avg("intentmind", "f1"), 3),
            "intentmind_avg_hit_at_k": round(avg("intentmind", "hit_at_k"), 3),
            "intentmind_avg_mrr": round(avg("intentmind", "mrr"), 3),
            "intentmind_avg_latency_ms": round(avg("intentmind", "latency_ms"), 3),
            "intentmind_p50_latency_ms": self._percentile([r["intentmind"]["latency_ms"] for r in results], 50),
            "intentmind_p95_latency_ms": self._percentile([r["intentmind"]["latency_ms"] for r in results], 95),
            "intentmind_avg_tokens": round(avg("intentmind", "tokens"), 3),
            "intentmind_avg_token_saving_pct": round(avg("intentmind", "token_saving_pct"), 3),
            "intentmind_avg_direct_count": round(avg("intentmind", "direct_count"), 3),
            "intentmind_avg_associated_count": round(avg("intentmind", "associated_count"), 3),
            "intentmind_latency_breakdown": {k: breakdown_avg(k) for k in breakdown_keys},
        }
        
        return {
            "summary": summary,
            "details": results
        }

    def generate_markdown_report(self, results: Dict[str, Any], title: str = "Real World Benchmark Report") -> str:
        summary = results["summary"]
        details = results["details"]
        wins = [
            item for item in details
            if item["intentmind"]["f1"] > item["classic_rag"]["f1"]
        ]
        losses = [
            item for item in details
            if item["intentmind"]["f1"] < item["classic_rag"]["f1"]
        ]
        ties = len(details) - len(wins) - len(losses)

        lines = [
            f"# {title}",
            "",
            "This report compares the same memory corpus with classic vector RAG and Intentmind.",
            "The goal is not to claim universal superiority, but to show where associative recall helps, where it fails, and why.",
            "",
            "## Summary",
            "",
            f"- Total queries: {summary['total_queries']}",
            f"- Top-K baseline: {summary['top_k']}",
            f"- Intentmind wins/ties/losses by F1: {len(wins)} / {ties} / {len(losses)}",
            "",
            "| System | Precision | Recall | F1 | Hit@K | MRR | Avg Tokens | p50 Latency | p95 Latency |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| Classic RAG | {summary['classic_rag_avg_precision']} | "
                f"{summary['classic_rag_avg_recall']} | {summary['classic_rag_avg_f1']} | "
                f"{summary['classic_rag_avg_hit_at_k']} | {summary['classic_rag_avg_mrr']} | "
                f"{summary['classic_rag_avg_tokens']} | {summary['classic_rag_p50_latency_ms']} ms | "
                f"{summary['classic_rag_p95_latency_ms']} ms |"
            ),
            (
                f"| Intentmind | {summary['intentmind_avg_precision']} | "
                f"{summary['intentmind_avg_recall']} | {summary['intentmind_avg_f1']} | "
                f"{summary['intentmind_avg_hit_at_k']} | {summary['intentmind_avg_mrr']} | "
                f"{summary['intentmind_avg_tokens']} | {summary['intentmind_p50_latency_ms']} ms | "
                f"{summary['intentmind_p95_latency_ms']} ms |"
            ),
            "",
            f"Average token saving: {summary['intentmind_avg_token_saving_pct']}%",
            f"Average direct/associated memories: {summary['intentmind_avg_direct_count']} / {summary['intentmind_avg_associated_count']}",
            "",
            "## Intentmind Latency Breakdown (avg ms)",
            "",
            "| Phase | Avg ms | Description |",
            "|---|---:|---|",
        ]

        lb = summary.get("intentmind_latency_breakdown", {})
        phase_desc = {
            "embed_query_ms": "Query embedding (SentenceTransformer or equivalent)",
            "emotion_ms": "Emotion detection",
            "extractor_ms": "Intent extraction (fixture / LLM)",
            "recall_ms": "Graph traversal + chunk scoring",
            "prompt_ms": "Prompt assembly",
        }
        for key in ["embed_query_ms", "emotion_ms", "extractor_ms", "recall_ms", "prompt_ms"]:
            desc = phase_desc.get(key, key)
            val = lb.get(key, 0.0)
            lines.append(f"| {key.replace('_ms', '')} | {val} | {desc} |")
        lines.extend([
            "",
            f"> **Note:** The extractor phase uses a fixture-backed dictionary lookup in this benchmark (<1ms). "
            f"With a real LLM extractor (e.g. GPT-4o-mini), this phase would add 200–800ms per query.",
            "",
            "## Why This Can Be Better Than Classic RAG",
            "",
            "Classic RAG retrieves chunks by query-to-chunk vector similarity. Intentmind first activates query intents, then traverses graph neighbors and scores linked chunks.",
            "This matters when the query does not repeat the exact missing concept, but touches an associated concept that was observed with it before.",
            "",
            "## Representative Associative Cases",
            "",
        ])

        associative_cases = [
            item for item in details
            if any(result.get("layer", 0) > 0 for result in item["intentmind"]["retrieved"])
        ][:8]
        if not associative_cases:
            lines.append("No associated-memory cases were returned in this run.")
        for item in associative_cases:
            lines.extend(self._case_lines(item))

        lines.extend(["", "## Failure Or Risk Cases", ""])
        risky = [
            item for item in details
            if item["intentmind"]["false_negatives"] or item["intentmind"]["false_positives"]
        ][:8]
        if not risky:
            lines.append("No Intentmind false positives or false negatives were observed in this run.")
        for item in risky:
            lines.extend(self._case_lines(item))

        lines.extend([
            "",
            "## Methodology Notes",
            "",
            "- Both systems use the same corpus and same embedder.",
            "- Classic RAG uses vector top-k chunk retrieval.",
            "- Intentmind uses faithful fixture extraction in this benchmark so recall dynamics are deterministic.",
            "- Public claims require larger datasets, independent ground truth, and repeated runs with median/p95 reporting.",
        ])
        return "\n".join(lines) + "\n"

    def _case_lines(self, item: Dict[str, Any]) -> List[str]:
        expected = ", ".join(item["expected_chunks"]) or "(none)"
        rag = ", ".join(r["chunk_id"] for r in item["classic_rag"]["retrieved"]) or "(none)"
        im = ", ".join(
            f"{r['chunk_id']}[{r.get('reason')}:{' > '.join(r.get('path', []))}]"
            for r in item["intentmind"]["retrieved"]
        ) or "(none)"
        reason = item["analysis"].get("expected_reason") or "No explicit reason provided."
        return [
            f"### {item['query']}",
            "",
            f"- Expected chunks: {expected}",
            f"- Classic RAG: {rag}",
            f"- Intentmind: {im}",
            f"- Expected reason: {reason}",
            "",
        ]

    def _precision_recall(self, retrieved: List[str], expected: set) -> tuple[float, float]:
        if not expected:
            return (1.0, 1.0) if not retrieved else (0.0, 1.0)
        if not retrieved:
            return 0.0, 0.0
        hits = len(set(retrieved).intersection(expected))
        precision = hits / len(retrieved)
        recall = hits / len(expected)
        return precision, recall

    def _f1(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _hit_at_k(self, retrieved: List[str], expected: set, top_k: int) -> int:
        if not expected:
            return 1 if not retrieved else 0
        return 1 if set(retrieved[:top_k]).intersection(expected) else 0

    def _mrr(self, retrieved: List[str], expected: set) -> float:
        if not expected:
            return 1.0 if not retrieved else 0.0
        for idx, chunk_id in enumerate(retrieved):
            if chunk_id in expected:
                return 1.0 / (idx + 1)
        return 0.0

    def _percentile(self, values: List[float], percentile: int) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        rank = ceil((percentile / 100) * len(ordered)) - 1
        rank = max(0, min(rank, len(ordered) - 1))
        return round(ordered[rank], 3)
