"""
News Data Ingestion Script
==============================
Reads the category_news_data.csv file and ingests 200 items into a new
Universal Cognitive Field memory called news_memory.json.
"""
import os
import sys
import time
import argparse
import csv
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from intentmind import IntentmindMemory
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to category_news_data.csv")
    parser.add_argument("--limit", type=int, default=200, help="Max news items to ingest")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between API calls")
    args = parser.parse_args()

    output_db = "news_memory.json"

    print("Initializing a fresh Intentmind Memory...")
    memory = IntentmindMemory()

    if not os.path.exists(args.file):
        print(f"Error: CSV file not found at {args.file}")
        return

    print(f"Loading {args.file}...")
    
    news_items = []
    with open(args.file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            news_items.append(row)
            if len(news_items) >= args.limit:
                break
    
    print(f"Loaded {len(news_items)} news articles.")
    
    model_name = os.getenv("OPENAI_MODEL")
    print(f"Using Model: {model_name}")

    print(f"\nIngesting {len(news_items)} news items into Universal Cognitive Field...")
    success = 0
    errors = 0

    for i, row in enumerate(tqdm(news_items, desc="Ingesting News")):
        # Combine short and long descriptions
        title = row.get("short_description", "").strip()
        body = row.get("long_description", "").strip()
        category = row.get("category", "UNKNOWN")
        source_id = row.get("id", str(i))
        
        text = f"{title}\n{body}"
        if not text.strip():
            continue

        metadata = {
            "category": category,
            "created_at": row.get("created_at", "")
        }

        try:
            # We add the category into the text so the LLM can easily infer the domain
            enriched_text = f"[Category: {category}]\n{text}"
            memory.add(enriched_text, source=f"news_{source_id}")
            success += 1
        except Exception as e:
            print(f"\nError on item {source_id}: {e}")
            errors += 1
            
        time.sleep(args.delay)

    print(f"\n--- Ingestion Complete ---")
    print(f"News items added: {success}")
    print(f"Errors: {errors}")

    stats = memory._store.stats()
    print(f"NEWS Graph: {stats['intents']} Nodes | {stats['edges']} Edges | {stats['chunks']} Chunks")

    print(f"\nSaving database as {output_db}...")
    memory.save(output_db)
    print("Done! You can now run the UI with this memory.")

if __name__ == "__main__":
    main()
