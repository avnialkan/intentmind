import json

data = json.load(open('memory.json', encoding='utf-8'))
intents = data.get('intents', [])
karavan_id = next((i['intent_id'] for i in intents if i['label'] == 'karavan'), None)

edges = data.get('edges', [])
if isinstance(edges, dict):
    edges = list(edges.values())

karavan_edges = [e for e in edges if e['source_id'] == karavan_id or e['target_id'] == karavan_id]
print(f'Karavan ID: {karavan_id}')
print(f'Edges Count: {len(karavan_edges)}')
for e in karavan_edges:
    source_label = next((i['label'] for i in intents if i['intent_id'] == e['source_id']), e['source_id'])
    target_label = next((i['label'] for i in intents if i['intent_id'] == e['target_id']), e['target_id'])
    print(f"- {source_label} -> {target_label} ({e['edge_type']}) [confidence={e['confidence']}, support={e['support_count']}, state={e['state']}]")
