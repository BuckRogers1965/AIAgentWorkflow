import os
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re

# readme
# add these libraries
#
# pip install PyMuPDF faiss-cpu numpy sentence-transformers
#

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_pdf_file(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def process_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith('.txt'):
            text = read_text_file(file_path)

            pattern = r'(?<=[.!?])\s*\n+'
            paragraphs = re.split(pattern, text)
            paragraphs = [para.strip() for para in paragraphs if para.strip()]

            documents.extend([para.strip() for para in paragraphs if para.strip()])  # Add non-empty paragraphs
        elif filename.endswith('.pdf'):
            text = read_pdf_file(file_path)
            #print (text)
            #paragraphs = text.split(".")  # Split by double newline (common paragraph separator)
            pattern = r'(?<=[.!?])\s*\n+'
            paragraphs = re.split(pattern, text)
            paragraphs = [para.strip() for para in paragraphs if para.strip()]
            
            documents.extend([para.strip() for para in paragraphs if para.strip()])  # Add non-empty paragraphs
    return documents

def create_rag_database(directory):
    # Process documents
    documents = process_documents(directory)
    
    # Create embeddings
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(documents)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    return index, documents

# Usage
directory = "docs/"
index, documents = create_rag_database(directory)

# Save the index and documents for later use
faiss.write_index(index, "rag_index.faiss")
np.save("document_chunks.npy", documents)

print(f"RAG database created with {len(documents)} document chunks.")
