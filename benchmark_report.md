# Real World Benchmark Report

This report compares the same memory corpus with classic vector RAG and Intentmind.
The goal is not to claim universal superiority, but to show where associative recall helps, where it fails, and why.

## Summary

- Total queries: 60
- Top-K baseline: 5
- Intentmind wins/ties/losses by F1: 45 / 9 / 6

| System | Precision | Recall | F1 | Hit@K | MRR | Avg Tokens | p50 Latency | p95 Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Classic RAG | 0.05 | 0.158 | 0.074 | 0.233 | 0.115 | 54.983 | 0.64 ms | 1.1 ms |
| Intentmind | 0.511 | 0.703 | 0.562 | 0.767 | 0.7 | 22.767 | 34.38 ms | 47.42 ms |

Average token saving: 58.934%
Average direct/associated memories: 1.667 / 0.35

## Intentmind Latency Breakdown (avg ms)

| Phase | Avg ms | Description |
|---|---:|---|
| embed_query | 0.05 | Query embedding (SentenceTransformer or equivalent) |
| emotion | 0.0 | Emotion detection |
| extractor | 0.49 | Intent extraction (fixture / LLM) |
| recall | 35.49 | Graph traversal + chunk scoring |
| prompt | 0.08 | Prompt assembly |

> **Note:** The extractor phase uses a fixture-backed dictionary lookup in this benchmark (<1ms). With a real LLM extractor (e.g. GPT-4o-mini), this phase would add 200–800ms per query.

## Why This Can Be Better Than Classic RAG

Classic RAG retrieves chunks by query-to-chunk vector similarity. Intentmind first activates query intents, then traverses graph neighbors and scores linked chunks.
This matters when the query does not repeat the exact missing concept, but touches an associated concept that was observed with it before.

## Representative Associative Cases

### I need the spare key before taking the car, where is it?

- Expected chunks: car_006
- Classic RAG: car_010, finance_004, car_003, noise_001, finance_006
- Intentmind: car_010[direct_match:car], car_001[neighbor_intent_reactivation:car > insurance]
- Expected reason: Direct spare key memory.

### What dashboard warning appeared after the cold drive?

- Expected chunks: car_007
- Classic RAG: home_004, finance_002, car_001, work_007, noise_004
- Intentmind: finance_009[neighbor_intent_reactivation:car > insurance], car_001[neighbor_intent_reactivation:car > insurance]
- Expected reason: Cold drive should map to tire pressure warning.

### Which car help plan expired and was related to bundled insurance discounts?

- Expected chunks: finance_009, car_009
- Classic RAG: car_001, travel_010, finance_009, noise_002, finance_004
- Intentmind: finance_004[direct_match:car], finance_009[neighbor_intent_reactivation:car > insurance], car_001[neighbor_intent_reactivation:car > insurance]
- Expected reason: Roadside assistance bridges car operations and insurance finance.

### For the Antalya family beach plan, what reservation and car dependency do I have?

- Expected chunks: travel_006, travel_001
- Classic RAG: travel_006, car_010, finance_004, finance_006, car_003
- Intentmind: travel_006[direct_match:Antalya], finance_006[direct_match:car], travel_001[direct_match:Antalya], car_001[neighbor_intent_reactivation:car > insurance]
- Expected reason: Antalya activates hotel and beach/rental-car memories.

### Before booking the London flights, what document and savings item should I remember?

- Expected chunks: travel_003, finance_008
- Classic RAG: finance_008, travel_003, travel_002, travel_001, travel_006
- Intentmind: finance_008[direct_match:Antalya], travel_006[neighbor_intent_reactivation:insurance > car]
- Expected reason: London links travel document and finance planning.

### Which software subscription and paid trial should I manage?

- Expected chunks: finance_002, finance_003
- Classic RAG: finance_003, travel_010, car_001, finance_005, car_008
- Intentmind: finance_003[direct_match:cancel subscription], travel_010[neighbor_intent_reactivation:car > insurance], car_001[neighbor_intent_reactivation:travel > car]
- Expected reason: Subscription management spans renewal and cancellation.

### Which bank card pays for hosting?

- Expected chunks: finance_006
- Classic RAG: noise_001, car_003, car_010, finance_004, finance_006
- Intentmind: noise_001[direct_match:car], car_001[neighbor_intent_reactivation:car > insurance]
- Expected reason: Direct payment method recall.

### Which appliance makes noise during the spin cycle?

- Expected chunks: home_001
- Classic RAG: car_001, dev_002, finance_004, work_004, health_004
- Intentmind: home_001[direct_match:spin cycle], dev_002[semantic_chunk_match:FAISS], finance_004[neighbor_intent_reactivation:noise > car], car_001[neighbor_intent_reactivation:noise > car]
- Expected reason: Direct home maintenance recall.


## Failure Or Risk Cases

### Before the long drive tomorrow, what car maintenance or paperwork should I check?

- Expected chunks: car_008, car_005, car_001, car_002
- Classic RAG: car_010, car_003, finance_004, noise_001, finance_006
- Intentmind: car_010[direct_match:car]
- Expected reason: A car trip should activate service, inspection, brakes, and insurance neighbors.

### Where did I park the vehicle near the blue elevator?

- Expected chunks: car_003
- Classic RAG: health_007, health_009, noise_005, car_004, health_001
- Intentmind: (none)
- Expected reason: Direct parking recall.

### Why did the highway drive make me worry about fuel?

- Expected chunks: car_004
- Classic RAG: travel_003, travel_002, travel_004, finance_001, finance_008
- Intentmind: travel_003[semantic_chunk_match:Antalya]
- Expected reason: Fuel issue linked to the car highway memory.

### I need the spare key before taking the car, where is it?

- Expected chunks: car_006
- Classic RAG: car_010, finance_004, car_003, noise_001, finance_006
- Intentmind: car_010[direct_match:car], car_001[neighbor_intent_reactivation:car > insurance]
- Expected reason: Direct spare key memory.

### What dashboard warning appeared after the cold drive?

- Expected chunks: car_007
- Classic RAG: home_004, finance_002, car_001, work_007, noise_004
- Intentmind: finance_009[neighbor_intent_reactivation:car > insurance], car_001[neighbor_intent_reactivation:car > insurance]
- Expected reason: Cold drive should map to tire pressure warning.

### Which car help plan expired and was related to bundled insurance discounts?

- Expected chunks: finance_009, car_009
- Classic RAG: car_001, travel_010, finance_009, noise_002, finance_004
- Intentmind: finance_004[direct_match:car], finance_009[neighbor_intent_reactivation:car > insurance], car_001[neighbor_intent_reactivation:car > insurance]
- Expected reason: Roadside assistance bridges car operations and insurance finance.

### Where are the vehicle registration and tax payment papers?

- Expected chunks: finance_005, car_010
- Classic RAG: finance_003, health_005, car_001, dev_009, home_009
- Intentmind: finance_005[direct_match:vehicle registration], finance_003[semantic_chunk_match:analytics tool], car_010[direct_match:black folder], health_005[semantic_chunk_match:food safety]
- Expected reason: Registration document and tax receipt are related but stored in different chunks.

### For the Antalya family beach plan, what reservation and car dependency do I have?

- Expected chunks: travel_006, travel_001
- Classic RAG: travel_006, car_010, finance_004, finance_006, car_003
- Intentmind: travel_006[direct_match:Antalya], finance_006[direct_match:car], travel_001[direct_match:Antalya], car_001[neighbor_intent_reactivation:car > insurance]
- Expected reason: Antalya activates hotel and beach/rental-car memories.


## Methodology Notes

- Both systems use the same corpus and same embedder.
- Classic RAG uses vector top-k chunk retrieval.
- Intentmind uses faithful fixture extraction in this benchmark so recall dynamics are deterministic.
- Public claims require larger datasets, independent ground truth, and repeated runs with median/p95 reporting.
