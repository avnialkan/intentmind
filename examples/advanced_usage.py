from intentmind import IntentmindMemory

def main():
    # Use FakeEmbedder for fast local testing, in production use default SentenceTransformer
    print("Initializing IntentmindMemory...")
    mem = IntentmindMemory(is_test=True)

    print("\nAdding memories...")
    mem.add("IntentGraph enerji tabanlı bir hafıza sistemidir.")
    mem.add("Intentmind memory consolidation yaparak gereksiz duplicate'leri birleştirir.")
    mem.add("Görselleştirme için networkx ve pyvis kullanıyoruz.")
    mem.add("Intentmind memory consolidation yaparak gereksiz duplicate'leri birleştirir.") # Duplicate
    
    print("\nConsolidating memory...")
    stats = mem.consolidate()
    print("Consolidation stats:", stats)

    print("\nQuerying...")
    res = mem.query("Enerji modeli nasıl çalışır?")
    print("Found direct memories:", res["memories"]["direct"])
    print("Trace:")
    for t in res["trace"]:
        print(f"  - Layer {t['layer']} ({t.get('intent', '')}): {t.get('text', t.get('reason'))[:50]}")

    print("\nSimulating passage of time (tick)...")
    tick_res = mem.tick(hours=48.0)
    print("Tick stats:", tick_res)

    print("\nGenerating visualization...")
    try:
        html_path = mem.visualize("intentmind_graph.html")
        print(f"Visualization saved to: {html_path}")
    except ImportError as e:
        print("Skipping visualization:", e)

    print("\nSaving and reloading memory...")
    mem.save("advanced_snapshot.json")
    mem2 = IntentmindMemory.load("advanced_snapshot.json", is_test=True)
    print(f"Reloaded chunks: {mem2._store.stats()['chunks']}")

if __name__ == "__main__":
    main()
