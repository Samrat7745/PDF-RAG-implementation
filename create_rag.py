"""Ingest PDFs and diagrams, then persist FAISS and the hybrid-search corpus."""

from __future__ import annotations

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf
from tqdm import tqdm

from helpers.image_extraction import DiagramExtractor
from helpers.image_to_description import DiagramDescriber
from helpers.rag_storage import DocumentCorpusStore


PROJECT_ROOT = Path(__file__).resolve().parent
PDF_DIRECTORY = PROJECT_ROOT / "pdfs"
DIAGRAM_DIRECTORY = PROJECT_ROOT / "extracted_diagrams"
DESCRIPTIONS_PATH = PROJECT_ROOT / "image_descriptions.json"
VECTORSTORE_DIRECTORY = PROJECT_ROOT / "vectorstore"
VISION_MODEL = "llava"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_or_create_image_descriptions() -> list[Document]:
    """Extract diagrams and create or load their searchable descriptions."""
    extractor = DiagramExtractor(str(PDF_DIRECTORY), str(DIAGRAM_DIRECTORY))
    extractor.extract_diagrams_by_captions()

    if DESCRIPTIONS_PATH.is_file():
        print(f"Loading diagram descriptions from {DESCRIPTIONS_PATH}...")
        with DESCRIPTIONS_PATH.open("r", encoding="utf-8") as description_file:
            entries = json.load(description_file)
        descriptions = [
            Document(
                page_content=entry["page_content"],
                metadata=entry.get("metadata", {}),
            )
            for entry in entries
        ]
    else:
        describer = DiagramDescriber(model_name=VISION_MODEL)
        image_paths = sorted(
            path for path in DIAGRAM_DIRECTORY.iterdir() if path.is_file()
        )
        descriptions = []
        for image_path in tqdm(image_paths, desc="Describing diagrams"):
            descriptions.append(
                Document(
                    page_content=describer.describe_diagram(str(image_path)),
                    metadata={"source": str(image_path), "type": "image_description"},
                )
            )

        serialized = [
            {"page_content": document.page_content, "metadata": document.metadata}
            for document in descriptions
        ]
        with DESCRIPTIONS_PATH.open("w", encoding="utf-8") as description_file:
            json.dump(serialized, description_file, indent=2, ensure_ascii=False)
        print(f"Saved diagram descriptions to {DESCRIPTIONS_PATH}")

    # FAISS metadata must point to image files even if launched outside the repo.
    for document in descriptions:
        source = document.metadata.get("source")
        if isinstance(source, str):
            source_path = Path(source)
            if not source_path.is_absolute():
                document.metadata["source"] = str((PROJECT_ROOT / source_path).resolve())

    return descriptions


def load_pdf_documents() -> list[Document]:
    """Extract embedded PDF text directly without requiring Tesseract OCR."""
    documents: list[Document] = []
    pdf_paths = sorted(PDF_DIRECTORY.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {PDF_DIRECTORY}")
    pages_without_text = 0

    for pdf_path in pdf_paths:
        with pymupdf.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                page_text = page.get_text("text").strip()
                if page_text:
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": str(pdf_path.resolve()),
                                "page": page_number,
                                "type": "pdf_page",
                            },
                        )
                    )
                else:
                    pages_without_text += 1

    if pages_without_text:
        print(
            f"Warning: skipped {pages_without_text} pages without embedded text. "
            "Scanned pages need OCR to be indexed."
        )

    if not documents:
        raise ValueError(
            "No embedded text was found in the PDFs. Scanned PDFs require OCR "
            "(for example, Tesseract) before they can be indexed."
        )
    return documents


def build_rag() -> None:
    """Create and save the dense index and its matching document corpus."""
    if not PDF_DIRECTORY.is_dir():
        raise FileNotFoundError(f"PDF directory not found: {PDF_DIRECTORY}")

    descriptions = load_or_create_image_descriptions()
    pdf_documents = load_pdf_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pdf_documents + descriptions)
    if not chunks:
        raise ValueError(f"No documents found to index in {PDF_DIRECTORY}")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    VECTORSTORE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIRECTORY))

    corpus_store = DocumentCorpusStore(
        VECTORSTORE_DIRECTORY / "document_chunks.json", PROJECT_ROOT
    )
    corpus_store.save(chunks)
    print(f"Saved FAISS index and {len(chunks)} chunks to {VECTORSTORE_DIRECTORY}")


if __name__ == "__main__":
    build_rag()
