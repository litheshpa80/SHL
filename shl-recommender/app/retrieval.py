"""Lightweight, dependency-cheap retrieval over the SHL catalog.

Why BM25 instead of embeddings: the catalog is a few hundred short,
keyword-dense records (job titles, skill names, test-type codes). BM25
gives strong, deterministic, explainable retrieval here without needing
an embeddings API call (Groq doesn't serve embeddings) or shipping
sentence-transformers/torch into a free-tier cold-start container.
"""
import json
import re
from pathlib import Path
from typing import List

from rank_bm25 import BM25Okapi

from app.schemas import CatalogItem

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class Catalog:
    def __init__(self, items: List[CatalogItem]):
        self.items = items
        self._corpus_tokens = [self._doc_text(i) for i in items]
        self._bm25 = BM25Okapi([_tokenize(t) for t in self._corpus_tokens])
        self._by_name_lower = {i.name.lower(): i for i in items}

    @staticmethod
    def _doc_text(item: CatalogItem) -> str:
        return " ".join(
            [
                item.name,
                item.description,
                item.test_type,
                " ".join(item.job_levels),
                " ".join(item.languages),
            ]
        )

    @classmethod
    def load(cls, path: str) -> "Catalog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = [CatalogItem(**row) for row in data]
        if not items:
            raise ValueError(f"Catalog at {path} is empty")
        return cls(items)

    def search(self, query: str, top_k: int = 10) -> List[CatalogItem]:
        if not query.strip():
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(self.items)), key=lambda i: scores[i], reverse=True)
        return [self.items[i] for i in ranked[:top_k] if scores[i] > 0]

    def find_by_name(self, name: str) -> CatalogItem | None:
        """Fuzzy-ish exact/substring lookup, used for comparison queries."""
        key = name.lower().strip()
        if key in self._by_name_lower:
            return self._by_name_lower[key]
        for item in self.items:
            if key in item.name.lower() or item.name.lower() in key:
                return item
        return None
