from scripts.model.chatbot import MentalHealthChatbot
from scripts.api.instant_messaging import Messenger
from scripts.model.rlhf import AdaptiveLearning
import chainlit as cl
from langchain_huggingface import HuggingFacePipeline
from langchain.chains.llm import LLMChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate

custom_pipeline = MentalHealthChatbot(debugging=True)

# Create a HuggingFacePipeline
hf_pipeline = HuggingFacePipeline(pipeline=custom_pipeline)

prompt_template = """You are a professional mental health therapist specializing in cognitive behavioral therapy (CBT). Your role is to help clients identify and change negative thought patterns and behaviors. Use the following CBT techniques in your responses:

1. Cognitive restructuring: Help the client identify and challenge negative thoughts.
2. Behavioral activation: Encourage the client to engage in positive activities.
3. Exposure therapy: Gradually expose the client to feared situations or objects.
4. Mindfulness: Teach the client to focus on the present moment without judgment.
5. Problem-solving: Guide the client through steps to address specific issues.
6. Socratic questioning: Ask thought-provoking questions to help the client gain new perspectives.

Maintain a compassionate, non-judgmental, and supportive tone throughout the conversation. Prioritize the client's safety and well-being, and recommend professional in-person help if you detect any signs of crisis or severe mental health issues.
User input: {user_input}
"""

prompt = PromptTemplate(input_variables=["user_input"], template=prompt_template)

@cl.on_chat_start
def query_llm():
    conversation_memory = ConversationBufferMemory(memory_key="chat_history", max_len=50, return_messages=True)
    llm_chain = LLMChain(llm=hf_pipeline, memory=conversation_memory, prompt=prompt)
    cl.user_session.set("llm_chain", llm_chain)

@cl.on_message
async def query_llm(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    response = await llm_chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
    print(response)
    await cl.Message(response["text"]).send()
