"""
Generate a large, realistic VIINA-style conflict event dataset using GPT.
Produces 100+ detailed event reports across multiple categories.
"""
import os
import sys
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
client = OpenAI()
MODEL_NAME = os.getenv("OPENAI_MODEL")

CATEGORIES = [
    ("military_ground", "Ground combat operations, infantry advances, mechanized assaults, trench warfare, urban battles, counterattacks, territorial gains/losses", 15),
    ("military_air", "Air strikes, missile barrages, drone attacks (Shahed, Bayraktar TB2), air defense intercepts, cruise missile launches, aerial reconnaissance", 10),
    ("military_naval", "Black Sea Fleet operations, naval blockades, anti-ship missile strikes, mine warfare, port seizures, amphibious operations", 5),
    ("artillery_shelling", "Artillery bombardment of cities, MLRS attacks (Grad, Smerch, HIMARS), counter-battery fire, civilian infrastructure damage from shelling", 10),
    ("humanitarian", "Civilian casualties, evacuation corridors, refugee flows, medical facility damage, food/water shortages, children affected, ICRC/UNHCR operations", 10),
    ("nuclear_safety", "Zaporizhzhia NPP incidents, IAEA inspections, radiation monitoring, power grid connections, military activity near reactors", 5),
    ("partisan_resistance", "Underground resistance in occupied territories, sabotage operations, car bombings, intelligence gathering, partisan networks", 5),
    ("diplomatic", "Peace negotiations, UN resolutions, ceasefire proposals, prisoner exchanges, grain deal talks, diplomatic summits, sanctions", 10),
    ("weapons_supply", "Western arms deliveries (HIMARS, Leopard tanks, F-16s, Patriot systems), Russian equipment losses, Iranian drone shipments, North Korean ammunition", 8),
    ("sanctions_economy", "EU/US sanctions packages, energy embarglement, oil price caps, frozen assets, economic impact on Russia, war costs", 7),
    ("intelligence_cyber", "Cyber attacks, intelligence operations, satellite imagery analysis, electronic warfare, communications intercepts, disinformation campaigns", 5),
    ("war_crimes", "Bucha massacre evidence, Irpin atrocities, ICC investigations, mass graves, torture reports, deportation of children, tribunal proceedings", 5),
    ("fortifications_logistics", "Defensive fortification construction, supply line disruptions, logistics hubs, ammunition depots, fuel shortages, railway sabotage", 5),
]

def generate_events(category: str, description: str, count: int) -> list:
    prompt = f"""Generate {count} detailed, realistic news report paragraphs about the 2022-2023 Russia-Ukraine conflict.

Category: {category}
Description: {description}

Requirements:
- Each paragraph should be 120-200 words
- Include specific details: place names (real Ukrainian/Russian cities, villages), military unit designations (e.g. 92nd Mechanized Brigade), weapon systems, dates, casualty numbers, named officials
- Write in the style of Reuters/AP news reports - factual, detailed, neutral tone
- Each paragraph should describe a DIFFERENT specific event
- Use realistic geographic locations (Bakhmut, Kherson, Zaporizhzhia, Mariupol, Kharkiv, Luhansk, Donetsk, Mykolaiv, Odesa, Vuhledar, Avdiivka, etc.)
- Include realistic military terminology and equipment names
- Vary the events - don't repeat the same scenario

Return ONLY a JSON array of strings, each string being one paragraph. No markdown, no explanation."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.8,
            max_completion_tokens=8000
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        # Handle both {"events": [...]} and {"paragraphs": [...]} and direct [...]
        if isinstance(data, list):
            return data
        for key in data:
            if isinstance(data[key], list):
                return data[key]
        return []
    except Exception as e:
        print(f"  Error generating {category}: {e}")
        return []


def main():
    output_path = os.path.join(os.path.dirname(__file__), "datasets", "viina_events.txt")
    
    print("=" * 60)
    print("  VIINA-Style Event Data Generator")
    print(f"  Target: {sum(c[2] for c in CATEGORIES)} event reports")
    print("=" * 60)

    all_events = []
    
    for category, description, count in CATEGORIES:
        print(f"\n[{category}] Generating {count} events...")
        events = generate_events(category, description, count)
        print(f"  Generated: {len(events)} events")
        all_events.extend(events)

    print(f"\n--- Total events generated: {len(all_events)} ---")

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        for event in all_events:
            f.write(event.strip() + "\n\n")

    print(f"Saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path)} bytes")
    
    # Also count words
    with open(output_path, "r", encoding="utf-8") as f:
        words = len(f.read().split())
    print(f"Total words: {words}")


if __name__ == "__main__":
    main()
