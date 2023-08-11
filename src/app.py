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

    topic = await cl.AskUserMessage(content="What is the topic of discussion today?").send()
    if topic:
        await cl.Message(
            content=f"Searching for research on: {topic['content']}",
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
