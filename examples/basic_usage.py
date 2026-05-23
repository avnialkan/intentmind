from intentmind import IntentmindMemory

mem = IntentmindMemory(is_test=True)

samples = [
    "Intentmind hafıza sistemi embedding tabanlı intent graph yapısı kullanır.",
    "Energy model intentlerin zamanla güçlenmesini ve zayıflamasını sağlar.",
    "Emotional detector recall davranışını doğrudan değiştirmez, sadece modüle eder.",
    "Activation score query similarity, node energy, edge energy ve penalty değerlerinden oluşur.",
    "Prompt builder direct memory, associated memory ve weak echo katmanlarını oluşturur.",
    "Recall engine aktive olmuş intentlerden ilgili chunklara gider.",
    "Adaptive pruning graph explosion ve recall pollution problemini azaltır.",
    "Consolidation engine duplicate memory parçalarını birleştirir ve reinforcement uygular.",
    "Humanoid robotik için token maliyeti düşük, hedefli ve hızlı recall gerekir.",
]
for text in samples:
    mem.add(text)

result = mem.query("Bunu merak ediyorum: robotik hafızada token maliyetini nasıl azaltırız?")
print(result["prompt"])
print("\nTRACE")
for row in result["trace"]:
    print(row)
