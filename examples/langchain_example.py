import json
from intentmind import IntentmindMemory
from intentmind.integrations.langchain import IntentmindRetriever

def main():
    print("Initializing IntentmindMemory...")
    mem = IntentmindMemory(is_test=True)
    
    print("Ingesting sample memory...")
    mem.add(
        "Energy model (enerji modeli) sayesinde Intentmind eski ve kullanılmayan "
        "bilgileri unutur, token tasarrufu sağlar."
    )
    
    print("Creating IntentmindRetriever...")
    retriever = IntentmindRetriever(memory=mem)
    
    query = "Enerji modeli neden token tasarrufu yapıyor?"
    print(f"\nQuerying LangChain Retriever: '{query}'")
    
    # Normally, you would use this retriever in a LCEL chain:
    # chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    # For this example, we just invoke it directly.
    
    docs = retriever.invoke(query)
    
    print(f"\nRetrieved {len(docs)} document(s).")
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(f"Content: {doc.page_content}")
        print("Metadata:")
        # Print metadata beautifully, but omit the verbose trace for the console output
        meta_to_print = {k: v for k, v in doc.metadata.items() if not k.startswith("_intentmind")}
        print(json.dumps(meta_to_print, indent=2, ensure_ascii=False))
        
        if "_intentmind_trace" in doc.metadata:
            print("\n(Note: _intentmind_trace and _intentmind_emotion are also injected into metadata for explainability!)")

if __name__ == "__main__":
    main()
