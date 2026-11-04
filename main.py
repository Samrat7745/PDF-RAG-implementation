"""Run interactive question answering over the previously built RAG index."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from helpers.hybrid_retriever import BM25Index, HybridRetriever
from helpers.multimodal_qa import MultimodalRAGAnswerer
from helpers.rag_storage import DocumentCorpusStore


PROJECT_ROOT = Path(__file__).resolve().parent
VECTORSTORE_DIRECTORY = PROJECT_ROOT / "vectorstore"
TEXT_MODEL = "llama3.2"
VISION_MODEL = "llava"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_retriever() -> HybridRetriever:
    """Load persisted FAISS and corpus artifacts and construct hybrid search."""
    faiss_path = VECTORSTORE_DIRECTORY / "index.faiss"
    if not faiss_path.is_file():
        raise FileNotFoundError(
            f"RAG index not found at {faiss_path}. Run 'python create_rag.py' first."
        )

    corpus_store = DocumentCorpusStore(
        VECTORSTORE_DIRECTORY / "document_chunks.json", PROJECT_ROOT
    )
    chunks = corpus_store.load()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        str(VECTORSTORE_DIRECTORY),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return HybridRetriever(
        vectorstore=vectorstore,
        sparse_index=BM25Index(chunks),
        k=4,
        candidate_k=20,
        dense_weight=0.5,
        sparse_weight=0.5,
    )


def paraphrase_prompt(question: str, llm: ChatOllama) -> str:
    """Correct the question's wording without changing or expanding its meaning."""
    response = llm.invoke(
        "Correct only the spelling, grammar, and punctuation in the question below. "
        "Keep the same meaning and all names and technical terms. Do not answer it, "
        "explain your edits, or add search instructions. Return only the corrected "
        f"question.\n\nQuestion: {question}"
    )
    rewritten_query = str(response.content).strip().strip('"\'')
    if not rewritten_query or len(rewritten_query) > max(len(question) * 3, 160):
        return question
    if any(marker in rewritten_query.casefold() for marker in ("search for context", "rewritten query:")):
        return question
    return rewritten_query


def answer_question(
    question: str,
    retriever: HybridRetriever,
    paraphrase_query: bool = True,
) -> str:
    """Optionally rewrite the search query, retrieve context, and answer."""
    text_llm = ChatOllama(model=TEXT_MODEL, temperature=0)
    search_query = paraphrase_prompt(question, text_llm) if paraphrase_query else question
    if paraphrase_query:
        print(f"Search query: {search_query}")

    answerer = MultimodalRAGAnswerer(
        llm=text_llm,
        vision_llm=ChatOllama(model=VISION_MODEL, temperature=0),
        max_images=7,
    )
    search_queries = [question]
    if search_query.casefold() != question.casefold():
        search_queries.append(search_query)
    source_documents = retriever.retrieve_queries(search_queries)
    answer = answerer.invoke(question, source_documents)

    print("\nAnswer:")
    print(answer)
    print("\nSources:")
    text_sources = MultimodalRAGAnswerer.answer_sources(source_documents)
    for index, document in enumerate(text_sources, start=1):
        source = document.metadata.get("source", "unknown")
        page = document.metadata.get("page")
        page_info = f", page {page}" if page is not None else ""
        print(f"[Source {index}] {source}{page_info}")
    return answer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about indexed PDFs.")
    parser.add_argument(
        "query",
        nargs="?",
        help="Question to ask; omit it to enter a question interactively.",
    )
    parser.add_argument(
        "--no-paraphrase",
        action="store_true",
        help="Use the original question as-is for retrieval.",
    )
    args = parser.parse_args()
    question = args.query or input("Question: ").strip()
    if not question:
        raise SystemExit("A non-empty question is required.")

    answer_question(
        question,
        load_retriever(),
        paraphrase_query=not args.no_paraphrase,
    )


if __name__ == "__main__":
    main()
