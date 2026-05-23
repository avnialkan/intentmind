# -*- coding: utf-8 -*-
"""
FakeEmbedder vs SentenceTransformer karsilastirmasi.
Ayni 3 bellek, ayni sorgu, farkli embedder.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from intentmind import IntentmindMemory
from intentmind.embeddings import cosine_similarity


def run_scenario(label, mem):
    print("\n" + "=" * 70)
    print("  %s" % label)
    print("=" * 70)

    texts = [
        ("mem_yemek", "D\u00FCn Antalyada g\u00FCzel bir yemek yemi\u015Ftik"),
        ("mem_benzin", "Antalyaya giderken araban\u0131n benzini bitti"),
        ("mem_sigorta", "Araban\u0131n sigortas\u0131n\u0131 yapt\u0131rmam gerekiyor"),
    ]

    for chunk_id, text in texts:
        cid = mem.add(text, chunk_id=chunk_id)
        print("  [%s] '%s'" % (cid, text))

    # Chunk sayisi
    chunk_count = len(mem._store.chunks)
    chunk_ids = list(mem._store.chunks.keys())
    print("\n  Chunk sayisi: %d -> %s" % (chunk_count, chunk_ids))

    if chunk_count < 3:
        # Hangileri merge oldu?
        for cid, chunk in mem._store.chunks.items():
            labels = []
            for iid in chunk.intent_ids:
                i = mem._store.intents.get(iid)
                if i:
                    labels.append(i.label)
            print("    [%s] intents=%s" % (cid, labels))

    # Embedder similarity matrix
    print("\n  Embedding Similarity Matrix:")
    sims = {}
    for i, (cid_a, text_a) in enumerate(texts):
        for j, (cid_b, text_b) in enumerate(texts):
            if j <= i:
                continue
            emb_a = mem.embedder.embed(text_a)
            emb_b = mem.embedder.embed(text_b)
            sim = cosine_similarity(emb_a, emb_b)
            sims[(cid_a, cid_b)] = sim
            print("    %s vs %s = %.4f %s" % (
                cid_a, cid_b, sim,
                "<-- MERGE RISKI (>0.92)" if sim > 0.92 else ""))

    # Intent graph
    gs = mem.graph_summary()
    print("\n  Intents (%d): %s" % (gs["total_intents"], [i["label"] for i in gs["intent_list"]]))
    print("  Edges (%d)" % gs["total_edges"])
    print("  Chunks (%d)" % gs["total_chunks"])

    # Query
    query = "Arabay\u0131 yar\u0131n servise g\u00F6t\u00FCrece\u011Fim \u00E7\u00FCnk\u00FC \u0130stanbula gitmem gerekiyor"
    print("\n  Sorgu: '%s'" % query)

    result = mem.query(query)
    trace = result["trace"]

    accepted = [t for t in trace if t["decision"] == "accepted"]
    print("\n  Kabul edilen chunklar (%d):" % len(accepted))
    for t in accepted:
        print("    [%s] score=%.4f layer=%d intent=%-12s called_by=%-10s reason=%s" % (
            t["chunk_id"], t["score"], t["layer"], t["intent"],
            str(t.get("called_by", "-")) if t.get("called_by") else "-",
            t.get("reason", "-")))
        if t.get("path"):
            print("           path: %s" % " -> ".join(t["path"]))

    print("\n  Prompt boyutu: %d karakter" % len(result["prompt"]))

    return chunk_count, len(accepted), sims


def main():
    # --- FakeEmbedder ---
    mem_fake = IntentmindMemory(is_test=True)
    fake_chunks, fake_accepted, fake_sims = run_scenario("FAKE EMBEDDER (Test)", mem_fake)

    # --- SentenceTransformer ---
    from intentmind.embeddings import SentenceTransformerEmbedder
    real_embedder = SentenceTransformerEmbedder()
    mem_real = IntentmindMemory(embedder=real_embedder)
    real_chunks, real_accepted, real_sims = run_scenario("SENTENCE TRANSFORMER (Gercek)", mem_real)

    # --- Karsilastirma ---
    print("\n" + "=" * 70)
    print("  SONUC KARSILASTIRMASI")
    print("=" * 70)

    print("\n  %-30s %-15s %-15s" % ("Metrik", "FakeEmbedder", "SentenceTransformer"))
    print("  " + "-" * 60)
    print("  %-30s %-15d %-15d" % ("Olusan chunk sayisi", fake_chunks, real_chunks))
    print("  %-30s %-15d %-15d" % ("Kabul edilen chunk (query)", fake_accepted, real_accepted))

    print("\n  Similarity Karsilastirmasi:")
    for key in fake_sims:
        fake_s = fake_sims[key]
        real_s = real_sims.get(key, 0)
        print("    %s vs %s:  Fake=%.4f  Real=%.4f  %s" % (
            key[0], key[1], fake_s, real_s,
            "<-- FARK BURADA!" if abs(fake_s - real_s) > 0.15 else ""))


if __name__ == "__main__":
    main()
