import numpy as np
import torch
import os
from sentence_transformers import SentenceTransformer
from scripts.utilities.faiss_utils import save_faiss_index, load_faiss_index
import faiss

# Example documents
class RAGSystem:
    def __init__(self, model_name="multi-qa-MiniLM-L6-cos-v1"):
        # Load pre-trained model and tokenizer
        self.model = SentenceTransformer(model_name)

    def _embed_docs(self, documents):
        # Convert documents to embeddings
        document_embeddings = self.model.encode(documents)
        return document_embeddings
    
    def load_embeddings(self, documents, embeddings_path):
        # Check if the embeddings file exists
        if not os.path.exists(embeddings_path):
            # Embed documents and save embeddings
            self.embeddings = self._embed_docs(documents)
            np.save(embeddings_path, self.embeddings)

        else:
            self.embeddings = np.load(embeddings_path)
    
    def create_index(self, index_path):
        # Create FAISS index
        if not os.path.exists(index_path):
            index = faiss.IndexFlatL2(self.embeddings.shape[1])
            index.add(self.embeddings)
            save_faiss_index(index, index_path)
        else:
            index = load_faiss_index(index_path)
        return index
    
    def retrieve_documents(self, query, index, documents, k=5):
        query_embedding = self._embed_docs([query])
        distances, indices = index.search(query_embedding, k)
        return [documents[i] for i in indices[0]]

