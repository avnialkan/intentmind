try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
except ImportError:
    raise ImportError("Please install langchain-core to use the IntentmindRetriever: pip install intentmind[langchain]")

from typing import List, Any
from pydantic import Field
from ..runtime import IntentmindMemory

class IntentmindRetriever(BaseRetriever):
    """
    LangChain BaseRetriever adapter for IntentmindMemory.
    
    This wraps the Intentmind memory runtime into a standard LangChain retriever,
    allowing it to be used directly in LangChain chains (e.g. create_retrieval_chain).
    """
    
    memory: Any = Field(description="The underlying IntentmindMemory instance.")
    include_trace_in_metadata: bool = Field(
        default=True, 
        description="Whether to include the trace and emotional state in the first document's metadata."
    )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        # Query Intentmind
        result = self.memory.query(query)
        items = result["memories"]["items"]
        
        docs = []
        for i, item in enumerate(items):
            metadata = {
                "chunk_id": item["chunk_id"],
                "score": item["score"],
                "layer": item["layer"],
                "intent": item["intent"],
                "intent_ids": item["intent_ids"],
                "source": item["source"],
                "path_strength": item["path_strength"],
                "called_by": item.get("called_by"),
                "reason": item.get("reason"),
                "path": item.get("path", []),
            }
            
            # Embed trace info in the top result for debugging in LC chains
            if i == 0 and self.include_trace_in_metadata:
                metadata["_intentmind_trace"] = result["trace"]
                metadata["_intentmind_emotion"] = result["emotion"]
                
            docs.append(Document(page_content=item["text"], metadata=metadata))
            
        return docs
