# -*- coding: utf-8 -*-
"""
Kullanicinin verdigi senaryo ile Intentmind vs Classic RAG karsilastirmasi.

Bellekler:
  1. "dun antalyada guzel bir yemek yemistik"
  2. "antalyaya giderken arabanin benzini bitti"
  3. "arabanin sigortasini yaptirmam gerekiyor"

Sorgu:
  "arabayi yarin servise goturecegim cunku istanbula gitmem gerekiyor"

Beklenen davranis:
  - "araba" intenti uyanir (direct match)
  - Komsu intentler aktive olur: sigorta, benzin, antalya
  - Ilgili chunklar cagrisimsal olarak cekilir
  - Classic RAG sadece vector benzerligine bakar
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

from intentmind import IntentmindMemory
from intentmind.embeddings import cosine_similarity


def separator(title):
    print("\n" + "=" * 70)
    print("  " + title)
    print("=" * 70)


def main():
    # ============================================================
    #  ADIM 1: Bellek Olusturma
    # ============================================================
    separator("ADIM 1: Bellekleri Ekliyoruz")

    mem = IntentmindMemory(is_test=True)

    memories = [
        ("mem_yemek", "D\u00FCn Antalyada g\u00FCzel bir yemek yemi\u015Ftik"),
        ("mem_benzin", "Antalyaya giderken araban\u0131n benzini bitti"),
        ("mem_sigorta", "Araban\u0131n sigortas\u0131n\u0131 yapt\u0131rmam gerekiyor"),
    ]

    for chunk_id, text in memories:
        cid = mem.add(text, chunk_id=chunk_id)
        extracted = mem._intent_engine.extract_intents(text)
        print("\n  [%s] '%s'" % (cid, text))
        print("    -> Cikarilan intentler: %s" % extracted)

    # ============================================================
    #  ADIM 2: Graph Yapisi
    # ============================================================
    separator("ADIM 2: Olusan Intent Graph")

    gs = mem.graph_summary()
    print("\n  Intent Nodes (%d):" % gs["total_intents"])
    for i in gs["intent_list"]:
        print("    [%s] %-15s energy=%.3f  src=%d" % (i["state"], i["label"], i["energy"], i["source_count"]))

    print("\n  Edges (%d):" % gs["total_edges"])
    for e in gs["edge_list"]:
        print("    %-12s <--(%.3f)--> %-12s  state=%-10s co_act=%d" % (
            e["source"], e["weight"], e["target"], e["state"], e["co_activation_count"]))

    print("\n  Chunks (%d):" % gs["total_chunks"])
    for cid, chunk in mem._store.chunks.items():
        intent_labels = []
        for iid in chunk.intent_ids:
            intent = mem._store.intents.get(iid)
            if intent:
                intent_labels.append(intent.label)
        print("    [%s] intents=%s" % (cid, intent_labels))

    # ============================================================
    #  ADIM 3: Sorgu
    # ============================================================
    separator("ADIM 3: Sorgu Atiliyor")

    query = "Arabay\u0131 yar\u0131n servise g\u00F6t\u00FCrece\u011Fim \u00E7\u00FCnk\u00FC \u0130stanbula gitmem gerekiyor"
    print("\n  Sorgu: '%s'" % query)

    # Sorgudan cikarilan intentler
    query_intents = mem._intent_engine.extract_intents(query)
    print("  Sorgudan cikarilan intentler: %s" % query_intents)

    # ============================================================
    #  ADIM 4: Intentmind Recall (Layer-Based)
    # ============================================================
    separator("ADIM 4: Intentmind Recall (Layer-Based)")

    result = mem.query(query)

    # Layer yapisi
    query_emb = mem.embedder.embed(query)
    emotional = mem._emotion.detect(query_emb)
    layers = mem._recall.build_layers(query_emb, emotional, query_intent_labels=query_intents)

    for layer_id, items in layers.items():
        if not items:
            continue
        print("\n  Layer %d:" % layer_id)
        for item in items:
            print("    %-15s score=%.4f  called_by=%-10s  reason=%-30s  path=[%s]" % (
                item["intent"].label,
                item["score"],
                str(item.get("called_by", "-")) if item.get("called_by") else "-",
                item.get("reason", "-"),
                " -> ".join(item.get("path", []))))

    # Kabul edilen chunklar
    print("\n  --- Kabul Edilen Chunklar ---")
    trace = result["trace"]
    for t in trace:
        if t["decision"] == "accepted":
            print("    [%s] score=%.4f layer=%d intent=%-12s called_by=%-10s reason=%s" % (
                t["chunk_id"], t["score"], t["layer"], t["intent"],
                str(t.get("called_by", "-")) if t.get("called_by") else "-",
                t.get("reason", "-")))
            if t.get("path"):
                print("           path: %s" % (" -> ".join(t["path"])))

    # Reddedilen chunklar
    rejected = [t for t in trace if t["decision"] == "rejected"]
    if rejected:
        print("\n  --- Reddedilen Chunklar ---")
        for t in rejected:
            print("    [%s] score=%.4f layer=%d reason=%s" % (
                t["chunk_id"], t["score"], t["layer"], t.get("reason", "-")))

    # ============================================================
    #  ADIM 5: Classic RAG (Sadece Cosine Similarity)
    # ============================================================
    separator("ADIM 5: Classic RAG (Sadece Cosine Similarity)")

    print("\n  Sorgu: '%s'" % query)
    print("\n  Tum chunk'lara cosine similarity:")

    rag_results = []
    for cid, chunk in mem._store.chunks.items():
        sim = cosine_similarity(query_emb, chunk.embedding)
        rag_results.append((cid, sim, chunk.text))
        print("    [%s] sim=%.4f  '%s'" % (cid, sim, chunk.text[:60]))

    rag_results.sort(key=lambda x: x[1], reverse=True)
    rag_top = rag_results[:3]  # top-3

    print("\n  Classic RAG Top-3:")
    for rank, (cid, sim, text) in enumerate(rag_top, 1):
        print("    %d. [%s] sim=%.4f" % (rank, cid, sim))

    # ============================================================
    #  ADIM 6: Karsilastirma
    # ============================================================
    separator("ADIM 6: Intentmind vs Classic RAG Karsilastirma")

    im_accepted = [t["chunk_id"] for t in trace if t["decision"] == "accepted"]
    rag_top_ids = [r[0] for r in rag_top]

    print("\n  Intentmind kabul ettikleri : %s" % im_accepted)
    print("  Classic RAG top-3         : %s" % rag_top_ids)

    # Intentmind'in getirip RAG'in getiremedikleri
    im_only = [c for c in im_accepted if c not in rag_top_ids]
    rag_only = [c for c in rag_top_ids if c not in im_accepted]
    both = [c for c in im_accepted if c in rag_top_ids]

    print("\n  Her ikisi de getirdi       : %s" % (both if both else "yok"))
    print("  Sadece Intentmind getirdi  : %s" % (im_only if im_only else "yok"))
    print("  Sadece Classic RAG getirdi : %s" % (rag_only if rag_only else "yok"))

    # ============================================================
    #  ADIM 7: Prompt Ciktisi
    # ============================================================
    separator("ADIM 7: LLM'e Gonderilecek Prompt")

    prompt = result["prompt"]
    print("\n%s" % prompt)

    # ============================================================
    #  ADIM 8: Ozet
    # ============================================================
    separator("ADIM 8: Sonuc Ozeti")

    print("""
  INTENTMIND NASIL CALISIYOR:

  1. Kullanici 3 bellek ekledi:
     - yemek + antalya
     - araba + benzin + antalya  
     - araba + sigorta

  2. Graph otomatik olusturuldu:
     - araba <-> sigorta (ayni cumlede gecti)
     - araba <-> benzin  (ayni cumlede gecti)
     - araba <-> antalya (komsuluk)
     
  3. Sorgu: "arabayi yarin servise goturecegim cunku istanbula gitmem gerekiyor"
     - "araba" intenti dogrudan uyanir (Layer 0 - Direct Match)
     - "araba"nin komsusu "sigorta" aktive olur (Layer 1 - Neighbor)
     - "araba"nin komsusu "benzin" aktive olur (Layer 1 - Neighbor)
     - Bu komsuluk sayesinde sigorta ve benzin chunklari da cekilir

  4. Classic RAG ise:
     - Sadece "arabayi servise goturecegim" cumlesinin vector benzerligine bakar
     - "sigorta" kelimesi sorguda gecmedigi icin sigorta chunk'ini kacirabilir
     - Graph bilgisi, komsuluk, enerji... hicbirini kullanmaz

  5. Sonuc:
     - Intentmind: CAGRISIMSAL hafiza (araba -> sigorta yolu)
     - Classic RAG: DUZLEMSEL hafiza (sadece benzer kelimeleri arar)
""")


if __name__ == "__main__":
    main()
