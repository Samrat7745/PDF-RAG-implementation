"""Persistence helpers for the document corpus used by hybrid retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document


class DocumentCorpusStore:
    """Store and reload the exact chunks indexed by the dense FAISS store."""

    def __init__(self, path: str | Path, project_root: str | Path):
        self._path = Path(path)
        self._project_root = Path(project_root).resolve()

    def save(self, documents: list[Document]) -> None:
        """Persist document text and metadata as UTF-8 JSON."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        serialized_documents = [
            {
                "page_content": document.page_content,
                "metadata": self._serialize_metadata(document.metadata),
            }
            for document in documents
        ]
        with self._path.open("w", encoding="utf-8") as corpus_file:
            json.dump(serialized_documents, corpus_file, ensure_ascii=False, indent=2)

    def load(self) -> list[Document]:
        """Load persisted chunks, restoring paths relative to the project root."""
        if not self._path.is_file():
            raise FileNotFoundError(
                f"Document corpus not found at {self._path}. "
                "Run build_rag.py to create the RAG index first."
            )

        with self._path.open("r", encoding="utf-8") as corpus_file:
            serialized_documents: list[dict[str, Any]] = json.load(corpus_file)

        return [
            Document(
                page_content=item["page_content"],
                metadata=self._deserialize_metadata(item.get("metadata", {})),
            )
            for item in serialized_documents
        ]

    def _serialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        serialized = dict(metadata)
        source = serialized.get("source")
        if isinstance(source, str):
            source_path = Path(source)
            if source_path.is_absolute():
                try:
                    serialized["source"] = source_path.resolve().relative_to(
                        self._project_root
                    ).as_posix()
                except ValueError:
                    serialized["source"] = str(source_path.resolve())
        return serialized

    def _deserialize_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        restored = dict(metadata)
        source = restored.get("source")
        if isinstance(source, str):
            source_path = Path(source)
            if not source_path.is_absolute():
                restored["source"] = str((self._project_root / source_path).resolve())
        return restored
