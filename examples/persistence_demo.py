from intentmind import IntentmindMemory

path = "memory_snapshot.json"
mem = IntentmindMemory(is_test=True)
mem.add("Intentmind energy based associative memory runtime olarak çalışır.")
mem.save(path)

restored = IntentmindMemory.load(path, is_test=True)
print(restored.graph_summary())
print(restored.query("energy memory nasıl çalışır?")["prompt"])
