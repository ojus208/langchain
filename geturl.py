
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


llm = ChatOllama(model='mistral')

vectorstore = Chroma(   
    persist_directory="./chroma_db_tools",
    embedding_function=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
)


retriever = vectorstore.as_retriever()

after_rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | llm
    | StrOutputParser()
)



print(after_rag_chain.invoke('write code to write shell command to write inside a file  using langchain tools '))