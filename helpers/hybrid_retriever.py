"""Hybrid dense and BM25 retrieval for LangChain document collections."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


class BM25Index:
    """Immutable BM25 index over documents, used for lexical candidate retrieval."""

    def __init__(self, documents: list[Document], k1: float = 1.5, b: float = 0.75):
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")

        self._documents = documents
        self._k1 = k1
        self._b = b
        self._tokens = [_tokenize(document.page_content) for document in documents]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        self._lengths = [len(tokens) for tokens in self._tokens]
        self._average_length = sum(self._lengths) / len(self._lengths) if documents else 0.0
        self._document_frequencies: Counter[str] = Counter()
        self._postings: dict[str, list[int]] = defaultdict(list)

        for document_id, frequencies in enumerate(self._term_frequencies):
            self._document_frequencies.update(frequencies.keys())
            for token in frequencies:
                self._postings[token].append(document_id)

    def search(self, query: str, limit: int) -> list[Document]:
        """Return the highest-scoring lexical matches, preserving corpus order on ties."""
        query_tokens = _tokenize(query)
        if not query_tokens or limit <= 0 or not self._documents:
            return []

        scores: dict[int, float] = defaultdict(float)
        document_count = len(self._documents)
        for token in query_tokens:
            matching_ids = self._postings.get(token, ())
            if not matching_ids:
                continue

            frequency = self._document_frequencies[token]
            inverse_document_frequency = math.log1p(
                (document_count - frequency + 0.5) / (frequency + 0.5)
            )
            for document_id in matching_ids:
                term_frequency = self._term_frequencies[document_id][token]
                length = self._lengths[document_id]
                length_normalization = self._k1 * (
                    1 - self._b + self._b * length / (self._average_length or 1.0)
                )
                scores[document_id] += inverse_document_frequency * (
                    term_frequency * (self._k1 + 1)
                    / (term_frequency + length_normalization)
                )

        ranked_ids = sorted(scores, key=lambda item: (-scores[item], item))[:limit]
        return [self._documents[document_id] for document_id in ranked_ids]


class HybridRetriever(BaseRetriever):
    """Combine FAISS, BM25, and optional query variants with RRF."""

    vectorstore: Any
    sparse_index: Any
    k: int = 4
    candidate_k: int = 20
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    rrf_constant: int = 60

    def retrieve_queries(self, queries: list[str]) -> list[Document]:
        """Retrieve for multiple query formulations and fuse rankings with RRF.

        Each query is independently processed by dense and sparse retrieval. The
        regular retriever call fuses those two rankings; this method applies RRF
        once more across the per-query hybrid rankings.
        """
        unique_queries = list(dict.fromkeys(query.strip() for query in queries if query.strip()))
        if not unique_queries:
            return []
        if len(unique_queries) == 1:
            return self.invoke(unique_queries[0])

        fused_scores: dict[str, float] = defaultdict(float)
        documents_by_key: dict[str, Document] = {}
        for query in unique_queries:
            ranked_documents = self.invoke(query)
            for rank, document in enumerate(ranked_documents, start=1):
                key = self._document_key(document)
                documents_by_key[key] = document
                fused_scores[key] += 1.0 / (self.rrf_constant + rank)

        ranked_keys = sorted(fused_scores, key=lambda key: -fused_scores[key])
        return [documents_by_key[key] for key in ranked_keys[: self.k]]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        if self.k <= 0 or self.candidate_k <= 0:
            return []
        if self.dense_weight < 0 or self.sparse_weight < 0:
            raise ValueError("Retrieval weights cannot be negative")
        if self.dense_weight + self.sparse_weight == 0:
            raise ValueError("At least one retrieval weight must be greater than zero")

        dense_documents = self.vectorstore.similarity_search(query, k=self.candidate_k)
        sparse_documents = self.sparse_index.search(query, self.candidate_k)
        fused_scores: dict[str, float] = defaultdict(float)
        documents_by_key: dict[str, Document] = {}

        for weight, ranked_documents in (
            (self.dense_weight, dense_documents),
            (self.sparse_weight, sparse_documents),
        ):
            for rank, document in enumerate(ranked_documents, start=1):
                key = self._document_key(document)
                documents_by_key[key] = document
                fused_scores[key] += weight / (self.rrf_constant + rank)

        ranked_keys = sorted(fused_scores, key=lambda key: -fused_scores[key])
        return [documents_by_key[key] for key in ranked_keys[: self.k]]

    @staticmethod
    def _document_key(document: Document) -> str:
        metadata = json.dumps(document.metadata, sort_keys=True, default=str)
        return f"{document.page_content}\0{metadata}"
