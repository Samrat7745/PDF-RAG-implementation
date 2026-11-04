# PDF-RAG-implementation
Implementing RAG using a PDF directory, FAISS and LLama 3.2(want to use an abliterated version of deepseek but now enough VRAM). A pipeline to convert Diagrams and images to description to be added in the RAG

## Hybrid retrieval

The RAG pipeline combines two retrieval strategies over the PDF chunks and diagram descriptions:

- **Dense retrieval:** FAISS with `all-MiniLM-L6-v2` finds semantically similar passages.
- **Lexical retrieval:** an in-project BM25 index finds passages containing matching terms, which is useful for names, acronyms, and exact phrases.
- **Fusion:** weighted reciprocal-rank fusion merges dense and BM25 rankings. When query paraphrasing is enabled, RRF also combines results for the original and paraphrased questions, returning the top four chunks.

The default dense and lexical weights are equal in `main.py`. Increase one weight to favor that retrieval method. The lexical index is built from the persisted chunk corpus when the query application starts.

## Build and run

The scripts are intentionally split into two steps:

1. Build the RAG data and save the FAISS index and its matching chunk corpus:

	```powershell
	python create_rag.py
	```

	This step extracts embedded PDF text with PyMuPDF (no Tesseract executable needed), extracts/describes diagrams as needed, splits the corpus, and writes the artifacts under `vectorstore/`. Scanned pages without an embedded text layer are skipped; OCR would be needed to index those pages.
2. Run the LLM query pipeline using those saved artifacts:

	```powershell
	python main.py "Explain the Kimi architectures."
	```

	Omit the question argument to be prompted interactively. Re-run `create_rag.py` after changing PDFs or diagram descriptions so the dense and lexical indexes stay in sync.

## Multimodal diagram answers

The standard `llama3.2` Ollama model is text-only. The pipeline currently uses the installed `llava` model for both diagram descriptions and image-aware answers because it supports image input and works with the current Ollama setup. To use Llama 3.2 Vision instead, update `VISION_MODEL` in both `main.py` and `create_rag.py` after resolving the local Ollama `mllama` runtime error.

```powershell
ollama pull llava
```

`llama3.2` handles query paraphrasing and ordinary text questions. `llava` is used when generating uncached diagram descriptions and for questions explicitly asking about a visual; those answers receive matching source images. Generated image descriptions help retrieval, but are not treated as answer evidence, and non-visual questions omit them to avoid irrelevant image noise.
