import os
import tempfile
import requests
from semanticscholar import SemanticScholar
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
from langchain.tools import Tool
from langchain.utilities import SerpAPIWrapper

class AgentCreator():

    def __init__(self) -> None:
        self.tools = []
        self.prompt = None
        self.documents = None
        self.topic = ""

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
    
    def findResearch(self, topic: str):
        sch = SemanticScholar()
        results = sch.search_paper(topic, limit=15, fields=['title', 'year', 'openAccessPdf'])
        print(f"Results Found: {len}")
        data = []
        count_open_access = 0 
        
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Temporary directory created: {tmpdir}")

            for item in results.items:
                if count_open_access >= 5:
                    break 
                
                if item.openAccessPdf:
                    response = requests.get(item.openAccessPdf['url'])
                    if response.status_code == 200:
                        pdf_content = response.content
                        pdf_filename = os.path.join(tmpdir, f"{item.title}.docx")
                        with open(pdf_filename, "wb") as pdf_file:
                            pdf_file.write(pdf_content)
                        print(f"Loaded: {item.title}")
                        count_open_access += 1
                    else:
                        print(f"Failed to download PDF from {item.title}")

            loader = DirectoryLoader(tmpdir, show_progress=True)
            self.documents = loader.load()
        
        self.topic = topic
            
    def createQA(self, llm: OpenAI, embeddings: OpenAIEmbeddings) -> RetrievalQA:
        
        splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap= 100, separators = ['\n', '\n\n'])
        splits = splitter.split_documents(self.documents)

        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        qa_chain = RetrievalQA.from_llm(llm,retriever=vectorstore.as_retriever())

        self.tools.append(
            Tool.from_function(
                func=qa_chain.run,
                name="Document Search",
                description=f"useful for when you need to access research on a specific question regarding {self.topic}",
                coroutine = qa_chain.arun,
                return_direct=True
            )
        )

if __name__=="__main__":
    creator = AgentCreator()
    creator.findResearch("Muscle Hypertrophy and Load Effects")
    print(len(creator.documents))