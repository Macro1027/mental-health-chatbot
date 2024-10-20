# Third-party imports
from fastapi import FastAPI, Form, Depends
from decouple import config
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from twilio.rest import Client
import logging

# Internal imports
from scripts.model.chatbot_api import ChatbotAPI
from scripts.whatsapp.utils import send_message
from scripts.model.rag import RAGSystem
from scripts.utilities.document_utils import DocumentProcessor
from scripts.utilities.faiss_utils import load_faiss_index

MODEL_NAME = "google/gemma-2b"
bot = ChatbotAPI(model=MODEL_NAME)
app = FastAPI()

# Paths to the necessary files
doc_path = "/Users/marcolee/Library/CloudStorage/OneDrive-HarrowInternationalSchoolHongKong/mental health chatbot/data/raw/The Body Keeps the Score.epub"
index_path = "data/processed/faiss_index.index"
embeddings_path = "/Users/marcolee/Library/CloudStorage/OneDrive-HarrowInternationalSchoolHongKong/mental health chatbot/data/processed/embeddings.npy"

doc_processor = DocumentProcessor()
documents = doc_processor.create_documents(doc_path)

rag = RAGSystem()
rag.load_embeddings(documents, embeddings_path)
index = rag.create_index(index_path)

@app.get("/")
async def home():
    return {"msg": "up & running"}

@app.post("/message")
async def reply(Body: str = Form()):
    # Perform RAG
    context = rag.retrieve_documents(Body, index, documents, k=3)
    context = ", ".join(context)

    # Ask API
    prompt = f"""DOCUMENT:
{context}

QUESTION:
{Body}

INSTRUCTIONS:
Answer the users QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT doesn’t contain the facts to answer the QUESTION return {'NONE'}"""
    payload = {"inputs": Body}

    chat_response = bot.query(payload)[0]['generated_text']

    # Store the conversation in the database
    send_message("+85260389682", chat_response)
    return ""