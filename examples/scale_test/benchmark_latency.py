import os
import sys
import time
import statistics

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from intentmind import IntentmindMemory

TEST_QUERIES = [
    "who is mr. darcy?",
    "what does elizabeth think of the military officers?",
    "why did they go to london?",
    "describe the bennet family dynamics",
    "how does pride affect relationships in the story?"
]

def main():
    print("Loading Intentmind Memory Engine for Latency Benchmark...")
    if os.path.exists("scale_test.json"):
        memory = IntentmindMemory.load("scale_test.json")
    else:
        memory = IntentmindMemory()
    
    nodes = memory._store.intents.values()
    edges = memory._store.edges.values()
    print(f"Graph Status -> Nodes: {len(nodes)} | Edges: {len(edges)}")
    
    if len(nodes) < 10:
        print("Warning: Graph is very small. Run bulk_ingest.py first for a realistic benchmark.")

    print("\nRunning Latency Tests...")
    latencies = []
    
    for query in TEST_QUERIES:
        start_time = time.time()
        # We use memory.query() which runs IntentEngine + RecallEngine + PromptBuilder
        result = memory.query(query)
        end_time = time.time()
        
        latency = end_time - start_time
        latencies.append(latency)
        
        print(f"\nQuery: '{query}'")
        print(f"Latency: {latency:.4f} seconds")
        print(f"Retrieved Memory Items: {len(result['memories']['items'])}")
        print(f"Retrieved Context Length: {len(result['prompt'])} chars")

    print("\n--- Benchmark Results ---")
    print(f"Total Queries: {len(TEST_QUERIES)}")
    print(f"Average Latency: {statistics.mean(latencies):.4f} seconds")
    print(f"Min Latency: {min(latencies):.4f} seconds")
    print(f"Max Latency: {max(latencies):.4f} seconds")

if __name__ == "__main__":
    main()
