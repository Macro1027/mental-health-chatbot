from scripts.model.chatbot import MentalHealthChatbot
from scripts.api.instant_messaging import Messenger
from scripts.model.rlhf import AdaptiveLearning
import chainlit as cl

bot = MentalHealthChatbot(debugging=True)

@cl.on_message
async def handle_message(message: cl.Message):
    user_message = message.content
    response = bot.ask(user_message)
    await cl.Message(response).send()

if __name__ == "__main__":
    cl.run()
