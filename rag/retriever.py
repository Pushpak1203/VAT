"""End-to-end legal RAG retrieval."""
from __future__ import annotations

from utils.schema import LegalEvidence
from rag.embedder import BGEEmbedder
from rag.reranker import BGEReranker
from rag.vector_store import VectorStore


DEFAULT_LEGAL_DOCUMENTS = [
    {
        "title": "EU VAT Directive - Place of supply of goods",
        "jurisdiction": "EU",
        "citation": "Council Directive 2006/112/EC, Title V",
        "text": "The place of supply rules determine where intra-Community supplies and domestic supplies of goods are taxed.",
    },
    {
        "title": "EU VAT Directive - Exemptions for intra-Community supplies",
        "jurisdiction": "EU",
        "citation": "Council Directive 2006/112/EC, Article 138",
        "text": "Member States shall exempt supplies of goods dispatched or transported to a destination outside their territory but within the Community under the conditions set by the Directive.",
    },
    {
        "title": "EU VAT Directive - Reverse charge",
        "jurisdiction": "EU",
        "citation": "Council Directive 2006/112/EC, Article 196",
        "text": "VAT is payable by the customer for specified supplies of services where the reverse charge mechanism applies.",
    },
    {
        "title": "UK VAT post-Brexit",
        "jurisdiction": "UK",
        "citation": "HMRC VAT guidance - imports and exports",
        "text": "Goods imported into Great Britain are subject to import VAT and exports may qualify for zero rating subject to evidence requirements.",
    },
    {
        "title": "EU One Stop Shop",
        "jurisdiction": "EU",
        "citation": "Council Directive 2006/112/EC, Articles 369a et seq.",
        "text": "The Union OSS scheme can simplify VAT reporting for qualifying cross-border B2C supplies within the EU.",
    },
]


class LegalRetriever:
    def __init__(self, settings):
        self.settings = settings
        self.embedder = BGEEmbedder(settings)
        self.reranker = BGEReranker(settings)
        self.store = VectorStore(settings.postgres_dsn, settings.vector_table, settings.embedding_dimension)
        self._seeded = False

    def _ensure_seeded(self):
        if self._seeded:
            return
        try:
            embeddings = self.embedder.embed([d["text"] for d in DEFAULT_LEGAL_DOCUMENTS])
            docs = []
            for d, e in zip(DEFAULT_LEGAL_DOCUMENTS, embeddings):
                item = dict(d)
                item["embedding"] = e.tolist()
                docs.append(item)
            self.store.insert(docs)
            self._seeded = True
        except Exception:
            # Keep the retriever usable in tests/offline mode with a tiny deterministic lexical corpus.
            for d in DEFAULT_LEGAL_DOCUMENTS:
                d.setdefault("embedding", [0.0] * self.settings.embedding_dimension)
            self.store.insert(DEFAULT_LEGAL_DOCUMENTS)
            self._seeded = True

    def retrieve(self, query: str) -> list[LegalEvidence]:
        self._ensure_seeded()
        try:
            q = self.embedder.embed([query])[0].tolist()
            candidates = self.store.similarity_search(q, self.settings.rag_top_k)
        except Exception:
            candidates = list(DEFAULT_LEGAL_DOCUMENTS)
        ranked = self.reranker.rerank(query, candidates, self.settings.rerank_top_k)
        return [
            LegalEvidence(
                title=d["title"],
                jurisdiction=d["jurisdiction"],
                citation=d["citation"],
                text=d["text"],
                score=float(d.get("rerank_score", d.get("score", 0.0))),
            )
            for d in ranked
        ]
