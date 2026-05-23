"""
Mixed-Domain Ingestion Script
==============================
Loads the existing Pride & Prejudice database (scale_test.json) 
and adds Frankenstein chunks into the SAME memory.
Saves as mixed_domain_test.json.
"""
import os
import sys
import time
import argparse
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from intentmind import IntentmindMemory
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="examples/scale_test/frankenstein.txt")
    parser.add_argument("--limit", type=int, default=200, help="Max chunks from new source")
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()

    base_db = "scale_test.json"
    output_db = "mixed_domain_test.json"

    if not os.path.exists(base_db):
        print(f"Error: {base_db} not found. Run bulk_ingest.py first.")
        return

    print(f"Loading existing Pride & Prejudice database ({base_db})...")
    memory = IntentmindMemory.load(base_db)
    stats = memory._store.stats()
    print(f"Existing Graph: {stats['intents']} Nodes | {stats['edges']} Edges | {stats['chunks']} Chunks")

    print(f"\nLoading {args.file}...")
    with open(args.file, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=300, overlap=50)
    print(f"Frankenstein split into {len(chunks)} chunks.")

    if args.limit > 0:
        chunks = chunks[:args.limit]
        print(f"Limiting to {args.limit} chunks.")

    model_name = os.getenv("OPENAI_MODEL")
    print(f"Using Model: {model_name}")

    print("\nIngesting Frankenstein into the same memory...")
    success = 0
    errors = 0

    for i, chunk in enumerate(tqdm(chunks, desc="Ingesting Frankenstein")):
        try:
            memory.add(chunk, source="frankenstein.txt")
            success += 1
        except Exception as e:
            print(f"\nError on chunk {i}: {e}")
            errors += 1
        time.sleep(args.delay)

    print(f"\n--- Ingestion Complete ---")
    print(f"Frankenstein chunks added: {success}")
    print(f"Errors: {errors}")

    stats = memory._store.stats()
    print(f"MIXED Graph: {stats['intents']} Nodes | {stats['edges']} Edges | {stats['chunks']} Chunks")

    print(f"\nSaving mixed database as {output_db}...")
    memory.save(output_db)
    print("Done!")

if __name__ == "__main__":
    main()
