import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
from dotenv import load_dotenv
from openai import OpenAI
from intentmind import IntentmindMemory

load_dotenv()

# Setup OpenAI Client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("Hata: .env dosyasinda OPENAI_API_KEY bulunamadi.")
    sys.exit(1)

client = OpenAI(api_key=api_key)
model = os.environ.get("OPENAI_MODEL") 

print("=" * 60)
print(f"  Intentmind Otonom Hafiza Test Simülasyonu")
print(f"  Model: {model}")
print("=" * 60)

print("\n[Sistem] Intentmind (Gercek Embedder ile) baslatiliyor... Bu birkac saniye surebilir.")
try:
    mem = IntentmindMemory(is_test=False) # Gercek SentenceTransformer!
except ImportError:
    print("\n[Uyari] 'sentence-transformers' yuklu degil. Hizli test icin FakeEmbedder kullaniliyor.")
    mem = IntentmindMemory(is_test=True)

# Simülasyon Senaryosu (Günler ve Mesajlar)
SCENARIO = [
    {
        "day": 1,
        "tick_hours": 0,
        "message": "Selam, bugun yeni bir araba aldim! Cok heyecanliyim ama kaskosu bekledigimden cok daha pahaliymis."
    },
    {
        "day": 2,
        "tick_hours": 24,
        "message": "Kopegim Tarcin dun geceden beri biraz rahatsiz, umarim onemli bir seyi yoktur."
    },
    {
        "day": 4,
        "tick_hours": 48,
        "message": "Hafta sonu Istanbul'dan Antalya'ya yola cikiyorum. Kafami dinlemeye cok ihtiyacim var."
    },
    {
        "day": 5,
        "tick_hours": 24,
        "message": "Sence Tarcini bu sicakta yanimda goturmem arabada sorun yaratir mi?"
    }
]

chat_history = [
    {"role": "system", "content": "Sen zeki ve empati yetenegi yuksek bir asistansin. Kullanicinin gecmis anilarini sana sistem saglayacak. Cevaplarini kisa, dogal ve baglami kullanarak ver."}
]

for step in SCENARIO:
    day = step["day"]
    tick_hours = step["tick_hours"]
    user_msg = step["message"]
    
    print(f"\n" + "=" * 50)
    print(f"  GUN {day} | {tick_hours} saat gecti")
    print("=" * 50)
    
    if tick_hours > 0:
        stats = mem.tick(hours=tick_hours)
        print(f"  [Intentmind] Zaman akti. Hafiza zayiflama ve konsolidasyon calisti.")
        
    print(f"\nKullanici: {user_msg}")
    
    # 1. Intentmind Query (Hafizayi uyar)
    print(f"  [Intentmind] Sorgu isleniyor...")
    recall_result = mem.query(user_msg)
    
    # 2. Intentmind Chunk Ingest (Yeni bilgiyi hafizaya al)
    mem.add(user_msg)
    
    # Hafiza ciktisini (prompt) goster
    memory_prompt = recall_result["prompt"]
    print(f"  [Intentmind] Cikarilan Intentler: {', '.join([t['intent'] for t in recall_result['trace'] if t['layer']==0])}")
    
    # Asil prompt'u hazirla
    enriched_prompt = f"{memory_prompt}\n\nKullanici Mesaji: {user_msg}"
    chat_history.append({"role": "user", "content": enriched_prompt})
    
    # Asistan Cevabi
    print(f"  [LLM] Cevap dusunuyor...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=chat_history,
            max_completion_tokens=200
        )
        assistant_reply = response.choices[0].message.content
        print(f"\nAsistan: {assistant_reply}")
        
        # Gercek sohbette sistem arka planda calisir, modele gonderdigimiz karmasik prompt yerine 
        # asil user mesajini tarihe ekleriz ki context penceresi dolmasin.
        chat_history[-1] = {"role": "user", "content": user_msg}
        chat_history.append({"role": "assistant", "content": assistant_reply})
        
    except Exception as e:
        print(f"\n[LLM Hatasi]: {e}")
        
    time.sleep(2) # Okunabilirlik icin ufak bir bekleme

print("\n" + "=" * 60)
print("  Simulasyon Tamamlandi. Iste Son Graph Durumu:")
print("=" * 60)
gs = mem.graph_summary()
print(f"Toplam Intent: {gs['total_intents']} | Toplam Baglanti (Edge): {gs['total_edges']} | Chunk: {gs['total_chunks']}")
top_intents = [i['label'] for i in gs['intent_list'][:5]]
print(f"En Guclu 5 Intent: {top_intents}")
