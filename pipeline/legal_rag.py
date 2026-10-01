"""Step 5: legal RAG facade."""
from rag.retriever import LegalRetriever


class LegalRAG:
    def __init__(self, settings):
        self.retriever = LegalRetriever(settings)

    def retrieve(self, query: str):
        return self.retriever.retrieve(query)
