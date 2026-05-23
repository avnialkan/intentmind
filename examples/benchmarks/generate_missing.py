import os, sys, json, time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
client = OpenAI()
MODEL = os.getenv("OPENAI_MODEL")

missing = [
    ("military_ground", 15), ("military_naval", 5), ("artillery_shelling", 10),
    ("nuclear_safety", 5), ("partisan_resistance", 5), ("weapons_supply", 8),
    ("sanctions_economy", 7), ("intelligence_cyber", 5), ("war_crimes", 5),
    ("fortifications_logistics", 5)
]

all_events = []
for cat, count in missing:
    print(f"Generating {cat} ({count})...")
    prompt = (
        f"Generate exactly {count} realistic Reuters-style news paragraphs about "
        f"the 2022-2023 Russia-Ukraine war, category: {cat}. "
        f"Each paragraph should be 120-180 words with specific Ukrainian/Russian locations, "
        f"military unit names, dates, equipment names, and casualty numbers. "
        f'Return as JSON object with key "events" containing an array of strings.'
    )
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.8,
            max_completion_tokens=8000
        )
        data = json.loads(r.choices[0].message.content)
        events = data.get("events", data.get("paragraphs", []))
        if not events:
            for key in data:
                if isinstance(data[key], list):
                    events = data[key]
                    break
        all_events.extend(events)
        print(f"  Got {len(events)}")
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(0.5)

output = os.path.join(os.path.dirname(__file__), "datasets", "viina_events.txt")
with open(output, "a", encoding="utf-8") as f:
    for ev in all_events:
        f.write(ev.strip() + "\n\n")

print(f"\nAppended {len(all_events)} events")
with open(output, "r", encoding="utf-8") as f:
    text = f.read()
    paras = [p for p in text.split("\n\n") if p.strip()]
    print(f"Total paragraphs: {len(paras)}")
    print(f"Total words: {len(text.split())}")
    print(f"File size: {len(text)} bytes")
