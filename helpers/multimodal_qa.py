"""Build multimodal prompts from retrieved text and source images."""

from __future__ import annotations

import base64
import mimetypes
import os
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage


_IMAGE_SOURCE_SUFFIX = re.compile(r"_p\d+_fig_\d+$", re.IGNORECASE)
_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)
_CITATION_PATTERN = re.compile(r"\[\d+\]")
_VISUAL_QUERY_TERMS = re.compile(
    r"\b(diagram|figure|chart|graph|image|picture|visual|flowchart|"
    r"illustrat\w*|depict\w*|draw\w*|show|describe)\b",
    re.IGNORECASE,
)


class MultimodalRAGAnswerer:
    """Answer text queries normally and use a vision model for visual questions."""

    def __init__(self, llm: Any, max_images: int = 3, vision_llm: Any | None = None):
        if max_images < 0:
            raise ValueError("max_images cannot be negative")
        self._llm = llm
        self._vision_llm = vision_llm or llm
        self._max_images = max_images

    def invoke(self, question: str, documents: list[Document]) -> str:
        """Synthesize retrieved evidence and attach images only for visual queries."""
        selected_documents = self._select_documents(question, documents)
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": self._build_context(question, selected_documents),
            }
        ]
        if self._is_visual_query(question):
            content.extend(self._image_content_blocks(selected_documents))
        model = self._vision_llm if self._is_visual_query(question) else self._llm
        response = model.invoke([HumanMessage(content=content)])
        return str(response.content)

    @staticmethod
    def answer_sources(documents: list[Document]) -> list[Document]:
        """Return text evidence, excluding generated captions and reference lists."""
        return [
            document
            for document in documents
            if document.metadata.get("type") != "image_description"
            and not MultimodalRAGAnswerer._looks_like_reference_list(
                document.page_content
            )
        ]

    @staticmethod
    def _looks_like_reference_list(text: str) -> bool:
        """Recognize short chunks dominated by bibliography links and citations."""
        return (
            len(_URL_PATTERN.findall(text)) >= 2
            and len(_CITATION_PATTERN.findall(text)) >= 2
        )

    @classmethod
    def _select_documents(
        cls, question: str, documents: list[Document]
    ) -> list[Document]:
        """Drop image-description noise unless the question explicitly asks visually."""
        if not cls._is_visual_query(question):
            return [
                document
                for document in documents
                if document.metadata.get("type") != "image_description"
            ]

        pdf_stems = {
            os.path.normcase(os.path.splitext(os.path.basename(source))[0])
            for document in documents
            if document.metadata.get("type") != "image_description"
            and isinstance((source := document.metadata.get("source")), str)
            and source.lower().endswith(".pdf")
        }
        if not pdf_stems:
            return documents

        selected = []
        for document in documents:
            if document.metadata.get("type") != "image_description":
                selected.append(document)
                continue
            source = document.metadata.get("source")
            if isinstance(source, str) and cls._image_source_stem(source) in pdf_stems:
                selected.append(document)
        return selected

    @staticmethod
    def _is_visual_query(question: str) -> bool:
        return bool(_VISUAL_QUERY_TERMS.search(question))

    @staticmethod
    def _image_source_stem(source: str) -> str:
        stem = os.path.splitext(os.path.basename(source))[0]
        return os.path.normcase(_IMAGE_SOURCE_SUFFIX.sub("", stem))

    @staticmethod
    def _build_context(question: str, documents: list[Document]) -> str:
        source_sections = []
        text_documents = MultimodalRAGAnswerer.answer_sources(documents)
        for index, document in enumerate(text_documents, start=1):
            metadata = document.metadata
            source_sections.append(
                f"<source id=\"{index}\" "
                f"type=\"{metadata.get('type', 'text')}\" "
                f"file=\"{metadata.get('source', 'unknown')}\" "
                f"page=\"{metadata.get('page', 'unknown')}\">\n"
                f"{document.page_content}\n"
                f"</source>"
            )
        context = "\n\n".join(source_sections)
        return (
            "You are answering one question from a retrieved document set. Read and "
            "synthesize ALL source sections together before responding. Produce one "
            "coherent answer, not a separate answer or summary for each source. Treat "
            "the PDF text sections as primary evidence. For visual questions, inspect "
            "the actual attached image directly; generated image-description text is "
            "not evidence. Relate each visual observation to its image source. Preserve "
            "the exact model/version named by each source: never transfer a feature "
            "between Kimi variants or other related models. For a broad family question, "
            "state shared facts and clearly distinguish version-specific facts. Resolve "
            "overlap by combining consistent facts. Do not infer causation or claim one "
            "feature enables another unless a source explicitly says so. Ignore bibliography "
            "entries when stating technical facts. If evidence conflicts or is insufficient, "
            "state that clearly. Do not invent details, repeat this prompt, or describe "
            "an image generically when the text answers the question. Cite supporting "
            "sources inline using [Source 1], [Source 2], etc.\n\n"
            f"Retrieved sources (consider the set as a whole):\n"
            f"{context or '(No matching documents were retrieved.)'}\n\n"
            f"Question to answer: {question}\n\n"
            "Return only the single synthesized answer."
        )

    def _image_content_blocks(self, documents: list[Document]) -> list[dict[str, Any]]:
        if self._max_images == 0:
            return []

        blocks: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        image_count = 0

        for document in documents:
            if document.metadata.get("type") != "image_description":
                continue
            source = document.metadata.get("source")
            if not isinstance(source, str) or not os.path.isfile(source):
                continue

            normalized_path = os.path.normcase(os.path.abspath(source))
            if normalized_path in seen_paths:
                continue
            seen_paths.add(normalized_path)

            mime_type, _ = mimetypes.guess_type(source)
            if not mime_type or not mime_type.startswith("image/"):
                continue

            with open(source, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode("ascii")
            blocks.append(
                {
                    "type": "text",
                    "text": (
                        f"Retrieved image from {source}. Analyze it only if it directly "
                        "shows evidence relevant to the question; do not guess from a "
                        "generic or unrelated image."
                    ),
                }
            )
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded_image}"
                    },
                }
            )
            image_count += 1
            if image_count >= self._max_images:
                break

        return blocks
