import chainlit as cl
import os
from langchain import LLMChain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.llms import OpenAI
from langchain.agents import load_tools
from langchain.tools import Tool
from pydantic import BaseModel, Field
from langchain.utilities import SerpAPIWrapper

class AgentCreator():

    def __init__(self) -> None:
        self.tools = []
        self.prompt = ''

    def createAgent(self, llm: ChatOpenAI) -> AgentExecutor:

        self.createAgentPrompt()
        llm_chain = LLMChain(llm=llm, prompt=self.prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=self.tools, verbose=True)

        memory = ConversationBufferMemory(memory_key="chat_history")

        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=self.tools, verbose=True, memory=memory
        )

        return agent_chain
    
    def createAgentPrompt(self) -> None:
        prefix = """Have a conversation with a human, you are going to assist them by answering their questions primarily with research-based answers. 
         As a last resort, you can search online for a suitable answer. You have access to the following tools:"""
        suffix = """Begin!"

        {chat_history}
        Question: {input}
        {agent_scratchpad}"""

        self.prompt = ZeroShotAgent.create_prompt(
            self.tools,
            prefix=prefix,
            suffix=suffix,
            input_variables=["input", "chat_history", "agent_scratchpad"],
        )
    
    def createSearch(self) -> None:
        search = SerpAPIWrapper()
        self.tools.append(
            Tool(
                name = "Current Search",
                func=search.run,
                description="useful for when you need answers that you couldn't find in Document Search."
            ),
        )
    
    def createQA(self, llm: OpenAI, embeddings: OpenAIEmbeddings) -> RetrievalQA:

        loader = DirectoryLoader('docs/', show_progress=True)
        data = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap= 100, separators = ['\n', '\n\n'])
        splits = splitter.split_documents(data)

        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        qa_chain = RetrievalQA.from_llm(llm,retriever=vectorstore.as_retriever())

        self.tools.append(
            Tool.from_function(
                func=qa_chain.run,
                name="Document Search",
                description="useful for when you need to access research on a specific question",
                coroutine = qa_chain.arun
            )
        )