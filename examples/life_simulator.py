import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import random
from dotenv import load_dotenv
from openai import OpenAI
from intentmind import IntentmindMemory

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Hata: .env dosyasinda OPENAI_API_KEY bulunamadi.")
    sys.exit(1)

client = OpenAI(api_key=api_key)
model = os.environ.get("OPENAI_MODEL")

print("============================================================")
print("  Intentmind: Life Simulator (Dual-Agent Engine)")
print(f"  Model: {model}")
print("============================================================\n")

print("[Sistem] Intentmind (Gercek Embedder ile) baslatiliyor...")
try:
    mem = IntentmindMemory(is_test=False)
except ImportError:
    print("[Uyari] 'sentence-transformers' bulunamadi. FakeEmbedder kullaniliyor.")
    mem = IntentmindMemory(is_test=True)

# -------------------------------------------------------------------
#  PERSONA (SIMULATED HUMAN) SETUP
# -------------------------------------------------------------------
persona_system_prompt = """
Senin adin Alex (veya Ali). 32 yasinda, Istanbul'da yasayan, uluslararasi bir sirkette calisan kdemli bir yazilim gelistiricisin.
Karakteristik ozelliklerin:
- Cok calisiyorsun ve genelde streslisin.
- Diller arasi gecis yapmayi seviyorsun. Cogu zaman Turkce konusuyorsun ama bazen aniden Ingilizce kelimeler veya tamami Ingilizce olan cumleler kuruyorsun.
- Hobilerin: Arabalar, kamp yapmak, teknoloji, muzik ve yemek yapmak.
- Hayatinda biri var (Selin). Onunla inisli cikisli bir iliskiniz var.
- Bir kopegin var (Tarcin).
- Surekli hayati sorgulayan, gelecekle ilgili endiseleri olan ama kucuk seylerden de mutlu olan birisin.

    Gorevin: Asistaninla konusurken hayatindaki yeni olaylari cok cok uzun, detayli ve gercekci bir sekilde anlatmak.
    - Sadece 1-2 kelimeyle veya tek bir cumleyle gecistirme. Adeta gunluk yazar gibi icini dok, cok detaya gir.
    - Farkli konulari birbiriyle bagla, aralara kurgusal cok fazla rastgele konu serpistir. (Ornegin: ofisteki projeden bahsederken aniden kamp macerani, oradan ingilizce bir teknoloji podcastini, oradan Selin ile tartismanizi anlat).
    - Asistanin ne cevap verecegini beklemeden kendi hayatini yasa ve anlat.
    - Cok uzun cumleler kur, hikayeler anlat. Farkli dilleri (Turkce, Ingilizce, eger biliyorsan biraz da Almanca veya Ispanyolca kelimeler) dogalca araya karistir.
    - Amacin sistemi gercekten zorlayacak, kavramsal (cognitive) devasa bir hafiza grafigi (knowledge graph) urettirmek!
"""

persona_history = [{"role": "system", "content": persona_system_prompt}]

# -------------------------------------------------------------------
#  ASSISTANT SETUP
# -------------------------------------------------------------------
assistant_system_prompt = """
Sen Alex'in kisisel, hafizali ve empati yetenegi yuksek zeki asistanisin.
Kullanicinin gecmis anilarini sistem sana '[MEMORY]' bloklari icinde verecek.
Amacin: Alex'in soylediklerini dinlemek, ona empati ile karsilik vermek ve EGER verilen hafiza bloklarinda mevcut durumla ilgili / cagrissim yapan eski anilar varsa bunlari kullanarak dogal bir sekilde eski konulara atifta bulunmak.
"""

assistant_history = [{"role": "system", "content": assistant_system_prompt}]

# -------------------------------------------------------------------
#  SIMULATION LOOP
# -------------------------------------------------------------------
TOTAL_ITERATIONS = 12

print("\n[Simulasyon Basliyor]\n")

for i in range(TOTAL_ITERATIONS):
    print("=" * 70)
    
    # 1. TIME TICK
    if i == 0:
        tick_hours = 0
        time_context = "Su an ilk kez tanisiyorsunuz. Kendini tanit ve hayatindaki mevcut durumdan bahset."
    else:
        # Rastgele 2 saat ile 14 gun arasi bir zaman atlamasi
        tick_hours = random.randint(2, 336)
        days = tick_hours // 24
        hours = tick_hours % 24
        time_str = f"{days} gun {hours} saat" if days > 0 else f"{hours} saat"
        time_context = f"Su an asistaninla konustuktan sonra aradan {time_str} gecti. Gecen zamanda ne oldu? Ne hissediyorsun? Farkli bir konuya gecebilir veya ayni konudan devam edebilirsin. Istersen yari ingilizce konus."
        
        print(f"  [Zaman Atlamasi] {time_str} gecti. Hafiza zayiflamasi (decay) uygulaniyor...", flush=True)
        mem.tick(hours=tick_hours)

    print(f"  --- ITERASYON {i+1}/{TOTAL_ITERATIONS} ---", flush=True)

    # 2. PERSONA SPEAKS
    persona_prompt = time_context
    persona_history.append({"role": "user", "content": persona_prompt})
    
    try:
        response_persona = client.chat.completions.create(
            model=model,
            messages=persona_history,
            max_completion_tokens=250,
            temperature=0.8
        )
        alex_msg = response_persona.choices[0].message.content
    except Exception as e:
        print(f"\n[Persona LLM Hatasi]: {e}")
        break

    persona_history.append({"role": "assistant", "content": alex_msg})
    
    print(f"\n👱‍♂️ ALEX:\n{alex_msg}\n", flush=True)
    
    # 3. INTENTMIND PROCESSING
    print("  [Intentmind] Alex'in soyledikleri isleniyor ve anilar taranir...")
    recall_result = mem.query(alex_msg)
    mem.add(alex_msg)
    
    memory_prompt = recall_result["prompt"]
    extracted_intents = [t['intent'] for t in recall_result['trace'] if t['layer']==0]
    print(f"  [Intentmind] Uyandirilan Temel Intentler (Layer 0): {', '.join(extracted_intents)}")
    
    # 4. ASSISTANT RESPONDS
    enriched_prompt = f"{memory_prompt}\n\nAlex (Kullanici): {alex_msg}"
    assistant_history.append({"role": "user", "content": enriched_prompt})
    
    try:
        response_assistant = client.chat.completions.create(
            model=model,
            messages=assistant_history,
            max_completion_tokens=250,
            temperature=0.7
        )
        assistant_reply = response_assistant.choices[0].message.content
    except Exception as e:
        print(f"\n[Assistant LLM Hatasi]: {e}")
        break
        
    # Tarihceyi sismemesi icin asil mesaji tut
    assistant_history[-1] = {"role": "user", "content": alex_msg}
    assistant_history.append({"role": "assistant", "content": assistant_reply})
    
    print(f"\n🤖 ASISTAN:\n{assistant_reply}\n", flush=True)
    
    time.sleep(2) # Okunabilirlik icin kisa mola

print("\n" + "=" * 70)
print("  Simulasyon Tamamlandi. Iste Son Graph Durumu:")
print("=" * 70)
gs = mem.graph_summary()
print(f"Toplam Intent: {gs['total_intents']} | Toplam Baglanti (Edge): {gs['total_edges']} | Toplam Bellek Pulu (Chunk): {gs['total_chunks']}")

if gs['total_intents'] > 0:
    top_intents = [i['label'] for i in gs['intent_list'][:10]]
    print(f"En Guclu 10 Kavram (Hubs): {top_intents}")
    mem.visualize("life_graph.html")
    print("\n[Grafik] Ag haritasi 'life_graph.html' dosyasina kaydedildi. Tarayicinda acabilirsin!")

print("\nSimulasyon motoru testi basariyla tamamlandi!")
