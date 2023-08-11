import os
import chainlit as cl
from agent import AgentCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

@cl.on_chat_start
async def setup():
    llm = OpenAI(temperature=0)
    chatllm = ChatOpenAI(temperature=0, openai_api_key= os.environ['OPENAI_API_KEY'])
    embeddings = OpenAIEmbeddings(openai_api_key= os.environ['OPENAI_API_KEY'])
    creator = AgentCreator()

    topic = await cl.AskUserMessage(content="Hello! What topic should I look for research for?").send()
    if topic:
        await cl.Message(
            content=f"I've found some research on {topic['content']}, ask me anything about it!",
        ).send()

    creator.findResearch(topic['content'])
    creator.createSearch()
    creator.createQA(llm, embeddings)
    
    cl.user_session.set("agent", creator.createAgent(chatllm))


@cl.on_message
async def main(message: str):
    
    agent = cl.user_session.get('agent')
    response = agent.run(message)

    await cl.Message(content=response).send()
