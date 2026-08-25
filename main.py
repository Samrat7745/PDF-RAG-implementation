import os

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from helpers.image_extraction import DiagramExtractor
from helpers.image_to_description import DiagramDescriber
from tqdm import tqdm
import json

## diagram cropper/separator for the describer to create descriptions
diagrams = DiagramExtractor("./pdfs/","./extracted_diagrams")
diagrams.extract_diagrams_by_captions()

## Image describer creates a description for each diagram/graph and ignores other types of images
Describer = DiagramDescriber()
image_descriptions = []
# Filter valid files first so tqdm knows the exact total count
image_files = [
    f for f in os.listdir(diagrams.output_dir) 
    if os.path.isfile(os.path.join(diagrams.output_dir, f))
]
if not os.path.exists("image_descriptions.json"):
    for image_name in tqdm(image_files, desc="Describing diagrams"):
        image_path = os.path.join(diagrams.output_dir, image_name)
        image_descriptions.append(
            Document(
                page_content=Describer.describe_diagram(image_path),
                metadata={"source": image_path, "type": "image_description"},
            )
        )
    docs_dict_list = [
        {"page_content": doc.page_content, "metadata": doc.metadata}
        for doc in image_descriptions
    ]
    with open("image_descriptions.json", "w", encoding="utf-8") as f:
        json.dump(docs_dict_list, f, indent=4)

    print("Saved image descriptions to image_descriptions.json")


loader = DirectoryLoader(
    "./pdfs/",
    glob="**/*.pdf",
    loader_cls=UnstructuredPDFLoader,
    loader_kwargs={
        "mode": "single",
        "strategy": "hi_res",
        "pdf_infer_table_structure": True
    }
)
pages = loader.load()

if os.path.exists("image_descriptions.json"):
    print("Loading existing image descriptions from JSON...")
    with open("image_descriptions.json", "r", encoding="utf-8") as f:
        loaded_dicts = json.load(f)
        
    # Reconstruct the LangChain Document objects
    image_descriptions = [
        Document(page_content=item["page_content"], metadata=item["metadata"])
        for item in loaded_dicts
    ]

# The directory where you want to save the files

index_directory = "./vectorstore"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Check if we already built and saved the database
if os.path.exists(index_directory):
    print(f"Found existing vector store at {index_directory}. Loading it directly...")
    vectorstore = FAISS.load_local(
        index_directory, 
        embeddings, 
        allow_dangerous_deserialization=True 
    )
else:
    print("No existing vector store found. Building it from scratch...")
    
    # 1. Load PDFs
    loader = DirectoryLoader(
        "./pdfs/",
        glob="**/*.pdf",
        loader_cls=UnstructuredPDFLoader,
        loader_kwargs={
            "mode": "single",
            "strategy": "hi_res",
            "pdf_infer_table_structure": True
        }
    )
    pages = loader.load()

    # 2. Split and Combine
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = pages + image_descriptions
    splits = text_splitter.split_documents(documents)

    # 3. Embed and Build
    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # 4. Save for next time
    vectorstore.save_local(index_directory)
    print(f"Vector store saved to {index_directory}")


llm = ChatOllama(model="llama3.2", temperature=0)

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a Senior Teacher with expertise in teaching English as a second language. You are tasked with creating a lesson plan for a group of intermediate-level students. The lesson should focus on improving their conversational skills, vocabulary, and grammar. Please provide a detailed lesson plan that includes objectives, activities, and assessment methods."),
#     ("user", "{input}")
# ])

retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,  # to get source documents with answers
    chain_type="stuff",            # concatenates retrieved docs into prompt
)

query = "Explain me Kimi architectures."

result = qa_chain.invoke({"query": query})

print("Answer:", result['result'])
print("\nSource Documents:")
for doc in result['source_documents']:
    print(f"- {doc.page_content}")