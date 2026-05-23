import os
import sys
import time
import argparse
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from intentmind import IntentmindMemory

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks of words."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Bulk Ingest text into Intentmind")
    parser.add_argument("--file", type=str, required=True, help="Path to text file")
    parser.add_argument("--limit", type=int, default=50, help="Max number of chunks to process (API cost safety)")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds to wait between API calls to avoid rate limits")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File {args.file} not found.")
        return

    print(f"Loading {args.file}...")
    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=300, overlap=50)
    print(f"Text split into {len(chunks)} chunks.")
    
    if args.limit and args.limit > 0:
        chunks = chunks[:args.limit]
        print(f"Limiting to first {args.limit} chunks for safety.")

    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    model_name = os.getenv("OPENAI_MODEL")
    print(f"Using Model: {model_name}")

    memory = IntentmindMemory(model=model_name)
    
    print("\nStarting ingestion...")
    success_count = 0
    error_count = 0
    
    for i, chunk in enumerate(tqdm(chunks, desc="Ingesting")):
        try:
            # We add some conversational context metadata for testing
            memory.add(chunk, source=args.file)
            success_count += 1
        except Exception as e:
            print(f"\nError on chunk {i}: {e}")
            error_count += 1
            
        # Rate limiting backoff
        time.sleep(args.delay)

    print("\n--- Ingestion Complete ---")
    print(f"Successfully added: {success_count} chunks")
    print(f"Errors: {error_count}")
    
    # Print some stats from the DB
    nodes = memory._store.intents.values()
    edges = memory._store.edges.values()
    print(f"Graph Status -> Nodes: {len(nodes)} | Edges: {len(edges)}")

    # Save the memory to disk so benchmark scripts can load it
    print("\nSaving database to disk (scale_test.json)...")
    memory.save("scale_test.json")
    print("Save complete.")

if __name__ == "__main__":
    main()
