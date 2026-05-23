import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI
from intentmind import IntentmindMemory

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Hata: .env dosyasinda OPENAI_API_KEY bulunamadi.")
    sys.exit(1)

client = OpenAI(api_key=api_key)
model = os.environ.get("OPENAI_MODEL")

print("============================================================")
print("  Intentmind: Corporate Negotiation Simulator")
print("  (Multi-Agent B2B Meeting with Cognitive Memory)")
print(f"  Model: {model}")
print("============================================================\n")

print("[Sistem] Intentmind Cognitive Memory baslatiliyor...")
mem = IntentmindMemory(is_test=False)

# -------------------------------------------------------------------
#  PERSONA 1: KLAUS (GERMANY - BUYER)
# -------------------------------------------------------------------
klaus_system_prompt = """
Senin adin Klaus. Almanya, Münih'te dev bir lojistik sirketinde (LogisTech GmbH) Satin Alma Mudurusun (CPO).
Karakteristik ozelliklerin:
- Cok disiplinli, net ve pazarlikta sertsin. Butce konusunda taviz vermeyi sevmezsin.
- Arada bir Almanca kelimeler (Genau, Sehr gut, Entschuldigung vb.) veya Ingilizce is terimleri (deadline, SLA, ROI) kullaniyorsun.
- Kisisel hayat: Hafta sonlari Alplerde doga yuruyusu (hiking) yapmayi seversin. 12 yasinda bir oglun var, adi Lukas. Evde bir de Golden Retriever kopeginiz var.
- Amacin: Karsi taraftan (Ayse, Turk yazilim sirketi) alacaginiz yeni nesil 'Yapay Zeka Destekli Rota Optimizasyon Yazilimi' icin en iyi fiyati ve destek paketini koparmak. 
- Her mesajinda konuyu hem is pazarligina hem de araya kucuk kisisel detaylara / havadan sudan konulara (small talk) bagla. Cok dogal bir is toplantisi simule et. Uzun ve gercekci yanitlar ver.
"""

klaus_history = [{"role": "system", "content": klaus_system_prompt}]

# -------------------------------------------------------------------
#  PERSONA 2: AYSE (TURKEY - SELLER)
# -------------------------------------------------------------------
ayse_system_prompt = """
Senin adin Ayse. Turkiye, Istanbul'da basarili bir B2B SaaS sirketinde (OptiRoute AI) Global Satis Direktorusun.
Karakteristik ozelliklerin:
- Ikna kabiliyeti yuksek, sicakkanli ama sirketinin kar marjini korumaya calisan zeki bir saticisin.
- Gorusmeleri genellikle Turkce yurutuyorsun ama uluslararasi musterin oldugu icin Ingilizce kelimeler de serpiyorsun.
- Kisisel hayat: Gecen ay Bodrum'da harika bir tatil yaptin ama su an Istanbul trafiginden ve is stresinden dolayi yorgunsun. Yeni tasindigin evin dekorasyonuyla ugrasiyorsun.
- Amacin: Klaus'un sirketine bu yazilimi satmak. Yazilimin degerini (maliyet dusurme, yapay zeka gucu) vurgula. Fiyati cok kirmadan anlasmayi bagla.
- Her mesajinda Klaus'un soylediklerine karsilik ver. Hem is pazarigini ilerlet hem de kisisel/samimi diyaloglari (small talk) devam ettir. Cok uzun ve dogal yanitlar ver.
"""

ayse_history = [{"role": "system", "content": ayse_system_prompt}]

# -------------------------------------------------------------------
#  SIMULATION LOOP
# -------------------------------------------------------------------
TOTAL_ITERATIONS = 1

print("\n[Toplanti Basliyor: Zoom Gorusmesi]\n")

# Start conversation
last_message = "Merhaba Ayse. Toplantiya katildigin icin tesekkurler. Münih'te bugun hava cok kapali, umarim Istanbul'da isler ve hava daha iyidir. Yeni yaziliminiz icin fiyat teklifinizi inceledik ama acikcasi butcemizin biraz uzerinde."
klaus_history.append({"role": "assistant", "content": last_message})
ayse_history.append({"role": "user", "content": last_message})

print(f"🇩🇪 KLAUS:\n{last_message}\n")
mem.add(last_message)

def get_memory_context(text):
    result = mem.query(text)
    items = result.get("memories", {}).get("items", [])
    if not items:
        return ""
    
    context = "\n[TOPLANTI HAFIZASI - GECMIS KONU DETAYLARI]\n"
    for item in items:
        chunk_text = item["text"]
        context += f"- {chunk_text}\n"
    return context + "\n"

for i in range(TOTAL_ITERATIONS):
    print("=" * 80)
    print(f"--- ITERASYON {i+1} ---")
    
    # -----------------------------
    # AYSE'S TURN
    # -----------------------------
    time.sleep(1)
    # Give Ayse the shared memory context
    memory_context = get_memory_context(last_message)
    if memory_context.strip():
        ayse_prompt = f"{memory_context}Klaus az once sunu soyledi:\n'{last_message}'\nBuna hem is hem de kisisel baglamda cok dogal, detayli ve uzun bir cevap ver."
    else:
        ayse_prompt = f"Klaus az once sunu soyledi:\n'{last_message}'\nBuna hem is hem de kisisel baglamda cok dogal, detayli ve uzun bir cevap ver."
    
    temp_ayse_history = ayse_history[:-1] + [{"role": "user", "content": ayse_prompt}]
    
    ayse_response = client.chat.completions.create(
        model=model,
        messages=temp_ayse_history,
        temperature=0.8
    )
    ayse_reply = ayse_response.choices[0].message.content
    
    print(f"\n🇹🇷 AYSE:\n{ayse_reply}\n")
    
    # Save to memory and history
    mem.add(ayse_reply)
    ayse_history.append({"role": "assistant", "content": ayse_reply})
    klaus_history.append({"role": "user", "content": ayse_reply})
    
    # -----------------------------
    # KLAUS'S TURN
    # -----------------------------
    time.sleep(1)
    memory_context = get_memory_context(ayse_reply)
    if memory_context.strip():
        klaus_prompt = f"{memory_context}Ayse az once sunu soyledi:\n'{ayse_reply}'\nBuna hem pazarlik hem de kisisel baglamda cok dogal, detayli ve uzun bir cevap ver."
    else:
        klaus_prompt = f"Ayse az once sunu soyledi:\n'{ayse_reply}'\nBuna hem pazarlik hem de kisisel baglamda cok dogal, detayli ve uzun bir cevap ver."

    temp_klaus_history = klaus_history[:-1] + [{"role": "user", "content": klaus_prompt}]

    klaus_response = client.chat.completions.create(
        model=model,
        messages=temp_klaus_history,
        temperature=0.8
    )
    klaus_reply = klaus_response.choices[0].message.content
    
    print(f"\n🇩🇪 KLAUS:\n{klaus_reply}\n")
    
    mem.add(klaus_reply)
    klaus_history.append({"role": "assistant", "content": klaus_reply})
    ayse_history.append({"role": "user", "content": klaus_reply})
    
    last_message = klaus_reply

print("=" * 80)
print("\n[Toplanti Sona Erdi]")

# Kognitif Agi Gorsellestir
html_path = "corporate_negotiation_graph.html"
mem.visualize(html_path)
print(f"\nToplanti hafizasi gorsellestirildi: {html_path}")
