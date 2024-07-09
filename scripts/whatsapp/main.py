# Third-party imports
from fastapi import FastAPI, Form, Depends
from decouple import config
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from twilio.rest import Client
import logging
from chatbot_api import ChatbotAPI

from utils import send_message

MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"
bot = ChatbotAPI(model=MODEL_NAME)
app = FastAPI()

@app.get("/")
async def index():
    return {"msg": "up & running"}

@app.post("/message")
async def reply(Body: str = Form()):
    # The generated text
    payload = {"inputs": Body, "task": "text-generation"}
    
    chat_response = bot.query(payload)[0]['generated_text']

    # Store the conversation in the database
    send_message("+85260389682", chat_response)
    return ""