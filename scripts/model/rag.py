import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
from scripts.utilities.faiss_utils import save_faiss_index
import faiss

# Example documents
class RAGSystem:
    def __init__(self, model_name="multi-qa-MiniLM-L6-cos-v1"):
        # Load pre-trained model and tokenizer
        self.model = SentenceTransformer(model_name)

    def embed_docs(self, documents):
        # Convert documents to embeddings
        document_embeddings = self.model.encode(documents)
        print("Embedded docs")  
        return document_embeddings
    
    def create_index(self, document_embeddings, index_path):
        # Create FAISS index
        index = faiss.IndexFlatL2(document_embeddings.shape[1])
        index.add(document_embeddings)
        save_faiss_index(index, index_path)
    
    def retrieve_documents(self, query, index, documents, k=5):
        query_embedding = self.embed_docs([query])
        distances, indices = index.search(query_embedding, k)
        return [documents[i] for i in indices[0]]

