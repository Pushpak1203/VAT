"""PostgreSQL + pgvector store with a dependency-free in-memory test fallback."""
from __future__ import annotations

import math
from typing import Iterable


class VectorStore:
    def __init__(self, dsn: str, table: str, dimension: int):
        self.dsn = dsn
        self.table = table
        self.dimension = dimension
        self._memory: list[dict] = []
        self._conn = None

    def connect(self):
        if self._conn is not None:
            return self._conn
        try:
            import psycopg2
            self._conn = psycopg2.connect(self.dsn)
            self.ensure_schema()
            return self._conn
        except Exception:
            return None

    def ensure_schema(self):
        if not self._conn:
            return
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id BIGSERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    jurisdiction TEXT NOT NULL,
                    citation TEXT NOT NULL,
                    text TEXT NOT NULL,
                    embedding vector({self.dimension})
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.table}_embedding_idx
                ON {self.table} USING hnsw (embedding vector_cosine_ops)
            """)
        self._conn.commit()

    def insert(self, documents: Iterable[dict]):
        docs = list(documents)
        if not docs:
            return
        if not self.connect():
            self._memory.extend(docs)
            return
        with self._conn.cursor() as cur:
            for d in docs:
                embedding = d["embedding"]
                cur.execute(
                    f"INSERT INTO {self.table}(title,jurisdiction,citation,text,embedding) VALUES (%s,%s,%s,%s,%s)",
                    (d["title"], d["jurisdiction"], d["citation"], d["text"], embedding),
                )
        self._conn.commit()

    def similarity_search(self, embedding: list[float], top_k: int = 10) -> list[dict]:
        if self.connect():
            with self._conn.cursor() as cur:
                cur.execute(
                    f"""SELECT title,jurisdiction,citation,text,1-(embedding <=> %s::vector) AS score
                        FROM {self.table}
                        ORDER BY embedding <=> %s::vector
                        LIMIT %s""",
                    (embedding, embedding, top_k),
                )
                rows = cur.fetchall()
                return [
                    {"title": r[0], "jurisdiction": r[1], "citation": r[2], "text": r[3], "score": float(r[4])}
                    for r in rows
                ]
        # cosine similarity fallback
        def cosine(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(x*x for x in b))
            return dot / (na * nb) if na and nb else 0.0
        ranked = []
        for d in self._memory:
            item = dict(d)
            item["score"] = cosine(embedding, d["embedding"])
            ranked.append(item)
        return sorted(ranked, key=lambda x: x["score"], reverse=True)[:top_k]
