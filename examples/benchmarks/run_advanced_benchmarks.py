import os
import sys
import json
from typing import Dict, List
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from intentmind import IntentmindMemory
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
client = OpenAI()
MODEL_NAME = os.getenv("OPENAI_MODEL")

def generate_answer(query: str, context: str) -> str:
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.

Context:
{context}

Question: {query}"""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content

def evaluate_granular(query: str, answer: str, context: str) -> Dict[str, float]:
    """
    RAGAS-style granular evaluation using LLM-as-a-judge.
    Scores Faithfulness, Answer Relevance, and Context Precision out of 5.
    """
    prompt = f"""You are an expert evaluator of RAG systems.
Evaluate the following Answer based on the Question and Context provided.
Score the following metrics from 1 to 5 (5 is best):

1. Faithfulness: Does the answer strictly use the context, or does it hallucinate information not present? (5 = entirely grounded in context, 1 = mostly hallucinated)
2. Answer Relevance: Does the answer directly address the user's question? (5 = perfect direct answer, 1 = completely misses the point)
3. Context Precision: Is the provided context highly relevant and concise, or is it filled with irrelevant noise? (5 = highly relevant and precise, 1 = filled with noise)

Context:
{context}

Question:
{query}

Answer:
{answer}

Respond ONLY with a JSON object in this format:
{{
    "faithfulness": 5,
    "answer_relevance": 4,
    "context_precision": 5,
    "reasoning": "Explanation of scores..."
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
        print(f"Eval Error: {e}")
        return {"faithfulness": 0, "answer_relevance": 0, "context_precision": 0, "reasoning": "Error"}

def main():
    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "scale_test.json")
    dataset_path = os.path.join(os.path.dirname(__file__), "datasets", "test_qa.json")
    
    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found. Please run bulk_ingest first.")
        return
        
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found. Creating a sample one...")
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump([
                {"query": "Who is Mr. Darcy's friend?"},
                {"query": "What happened to Wickham?"},
                {"query": "Describe the overall progression of Elizabeth and Darcy's relationship."}
            ], f, indent=4)
        print("Sample dataset created. Run again.")
        return

    print("Loading Memory Engine...")
    memory = IntentmindMemory.load(db_path)
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    results = []
    
    print(f"Running Advanced Benchmarks on {len(queries)} queries...")
    for q_data in queries:
        query = q_data["query"]
        print(f"\n--- Testing Query: '{query}' ---")
        
        # Intentmind (GraphRAG)
        im_res = memory.query(query)
        im_context = im_res["prompt"]
        im_ans = generate_answer(query, im_context)
        im_eval = evaluate_granular(query, im_ans, im_context)
        
        # Baseline RAG (Vector)
        query_emb = memory.embedder.embed(query)
        raw_chunks = [c for _, c in memory._store.search_chunks(query_emb, top_k=5)]
        base_context = "\\n".join([f"- {c.text}" for c in raw_chunks])
        base_ans = generate_answer(query, base_context)
        base_eval = evaluate_granular(query, base_ans, base_context)
        
        print(f"Intentmind Scores -> Faithfulness: {im_eval['faithfulness']}, Relevance: {im_eval['answer_relevance']}, Context Precision: {im_eval['context_precision']}")
        print(f"Baseline Scores   -> Faithfulness: {base_eval['faithfulness']}, Relevance: {base_eval['answer_relevance']}, Context Precision: {base_eval['context_precision']}")
        
        results.append({
            "query": query,
            "intentmind": {"answer": im_ans, "eval": im_eval},
            "baseline": {"answer": base_ans, "eval": base_eval}
        })
        
    # Print Summary
    print("\\n=== Benchmark Summary ===")
    im_faith = sum(r['intentmind']['eval'].get('faithfulness', 0) for r in results) / len(results)
    im_rel = sum(r['intentmind']['eval'].get('answer_relevance', 0) for r in results) / len(results)
    im_prec = sum(r['intentmind']['eval'].get('context_precision', 0) for r in results) / len(results)
    
    base_faith = sum(r['baseline']['eval'].get('faithfulness', 0) for r in results) / len(results)
    base_rel = sum(r['baseline']['eval'].get('answer_relevance', 0) for r in results) / len(results)
    base_prec = sum(r['baseline']['eval'].get('context_precision', 0) for r in results) / len(results)
    
    print(f"Intentmind (GraphRAG) Averages - Faithfulness: {im_faith:.2f}, Relevance: {im_rel:.2f}, Precision: {im_prec:.2f}")
    print(f"Baseline (Vector) Averages   - Faithfulness: {base_faith:.2f}, Relevance: {base_rel:.2f}, Precision: {base_prec:.2f}")

if __name__ == "__main__":
    main()
