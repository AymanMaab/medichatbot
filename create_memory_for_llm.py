import os
from pathlib import Path
from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file and returns a list of LangChain Document objects.
    """
    documents = []
    try:
        reader = PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(Document(
                    page_content=text,
                    metadata={"page": i, "source": os.path.basename(pdf_path)}
                ))
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    return documents

def load_pdfs(data_dir):
    """
    Loads all PDF files from the specified directory and extracts text.
    """
    all_docs = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                docs = extract_text_from_pdf(full_path)
                all_docs.extend(docs)
    print(f"Total PDF pages loaded: {len(all_docs)}")
    return all_docs

def create_chunks(documents, chunk_size=500, chunk_overlap=50):
    """
    Splits documents into smaller text chunks using a recursive character splitter.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def get_embedding_model():
    """
    Initializes and returns a Hugging Face embedding model.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # Set to "cuda" if using GPU
    )

def main():
    documents = load_pdfs(DATA_PATH)

    non_empty_docs = [doc for doc in documents if doc.page_content.strip()]
    #print(f"Non-empty pages: {len(non_empty_docs)}")

    if not non_empty_docs:
        raise ValueError("No valid text found in PDFs.")

    text_chunks = create_chunks(non_empty_docs)
    #print(f"Total text chunks created: {len(text_chunks)}")

    if not text_chunks:
        raise ValueError("No text chunks to embed. Please check your PDF content.")

    embedding_model = get_embedding_model()
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    #print(f"FAISS vectorstore saved to: {DB_FAISS_PATH}")

if __name__ == "__main__":
    main()
# This script extracts text from PDF files, splits the text into chunks, and creates a FAISS vectorstore for efficient retrieval.
# It uses the Hugging Face embedding model for generating embeddings of the text chunks.