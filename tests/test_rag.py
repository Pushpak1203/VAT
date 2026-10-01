from config import Settings
from rag.vector_store import VectorStore


def test_in_memory_vector_store():
    store = VectorStore("invalid://dsn", "test_vectors", 3)
    store.insert([
        {"title": "A", "jurisdiction": "EU", "citation": "A1", "text": "VAT goods", "embedding": [1, 0, 0]},
        {"title": "B", "jurisdiction": "EU", "citation": "B1", "text": "services", "embedding": [0, 1, 0]},
    ])
    results = store.similarity_search([1, 0, 0], top_k=1)
    assert results[0]["title"] == "A"
