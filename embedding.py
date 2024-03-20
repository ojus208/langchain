
from pyexpat import model
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import CharacterTextSplitter

model_local = ChatOllama(model="mistral")

# 1. Split data into chunks
urls = [
'    https://python.langchain.com/docs/integrations/tools/alpha_vantage',
'	https://python.langchain.com/docs/integrations/tools/zapier',
'	https://python.langchain.com/docs/integrations/tools/youtube',
'	https://python.langchain.com/docs/integrations/tools/you',
'	https://python.langchain.com/docs/integrations/tools/yahoo_finance_news',
'	https://python.langchain.com/docs/integrations/tools/wolfram_alpha',
'	https://python.langchain.com/docs/integrations/tools/wikipedia',
'	https://python.langchain.com/docs/integrations/tools/wikidata',
'	https://python.langchain.com/docs/integrations/tools/twilio',
'	https://python.langchain.com/docs/integrations/tools/tavily_search',
'	https://python.langchain.com/docs/integrations/tools/stackexchange',
'	https://python.langchain.com/docs/integrations/tools/sql_database',
'	https://python.langchain.com/docs/integrations/tools/serpapi',
'	https://python.langchain.com/docs/integrations/tools/semanticscholar',
'	https://python.langchain.com/docs/integrations/tools/searx_search',
'	https://python.langchain.com/docs/integrations/tools/searchapi',
'	https://python.langchain.com/docs/integrations/tools/search_tools',
'	https://python.langchain.com/docs/integrations/tools/sceneXplain',
'	https://python.langchain.com/docs/integrations/tools/requests',
'	https://python.langchain.com/docs/integrations/tools/reddit_search',
'	https://python.langchain.com/docs/integrations/tools/python',
'	https://python.langchain.com/docs/integrations/tools/pubmed',
'	https://python.langchain.com/docs/integrations/tools/polygon',
'	https://python.langchain.com/docs/integrations/tools/passio_nutrition_ai',
'	https://python.langchain.com/docs/integrations/tools/openweathermap',
'	https://python.langchain.com/docs/integrations/tools/nuclia',
'	https://python.langchain.com/docs/integrations/tools/memorize',
'	https://python.langchain.com/docs/integrations/tools/lemonai',
'	https://python.langchain.com/docs/integrations/tools/ionic_shopping',
'	https://python.langchain.com/docs/integrations/tools/ifttt',
'	https://python.langchain.com/docs/integrations/tools/human_tools',
'	https://python.langchain.com/docs/integrations/tools/huggingface_tools',
'	https://python.langchain.com/docs/integrations/tools/graphql',
'	https://python.langchain.com/docs/integrations/tools/gradio_tools',
'	https://python.langchain.com/docs/integrations/tools/google_trends',
'	https://python.langchain.com/docs/integrations/tools/google_serper',
'	https://python.langchain.com/docs/integrations/tools/google_search',
'	https://python.langchain.com/docs/integrations/tools/google_scholar',
'	https://python.langchain.com/docs/integrations/tools/google_places',
'	https://python.langchain.com/docs/integrations/tools/google_lens',
'	https://python.langchain.com/docs/integrations/tools/google_jobs',
'	https://python.langchain.com/docs/integrations/tools/google_finance',
'	https://python.langchain.com/docs/integrations/tools/google_drive',
'	https://python.langchain.com/docs/integrations/tools/google_cloud_texttospeech',
'	https://python.langchain.com/docs/integrations/tools/golden_query',
'	https://python.langchain.com/docs/integrations/tools/filesystem',
'	https://python.langchain.com/docs/integrations/tools/exa_search',
'	https://python.langchain.com/docs/integrations/tools/eleven_labs_tts',
'	https://python.langchain.com/docs/integrations/tools/edenai_tools',
'	https://python.langchain.com/docs/integrations/tools/e2b_data_analysis',
'	https://python.langchain.com/docs/integrations/tools/ddg',
'	https://python.langchain.com/docs/integrations/tools/dataforseo',
'	https://python.langchain.com/docs/integrations/tools/dalle_image_generator',
'	https://python.langchain.com/docs/integrations/tools/connery',
'	https://python.langchain.com/docs/integrations/tools/chatgpt_plugins',
'	https://python.langchain.com/docs/integrations/tools/brave_search',
'	https://python.langchain.com/docs/integrations/tools/bing_search',
'	https://python.langchain.com/docs/integrations/tools/bearly',
'	https://python.langchain.com/docs/integrations/tools/bash',
'	https://python.langchain.com/docs/integrations/tools/awslambda',
'	https://python.langchain.com/docs/integrations/tools/arxiv',
'	https://python.langchain.com/docs/integrations/tools/apify',
'	https://python.langchain.com/docs/integrations/tools/alpha_vantage'
]
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
doc_splits = text_splitter.split_documents(docs_list)




# # 2. Convert documents to Embeddings and store them
vectorstore = Chroma.from_documents(
    persist_directory="./chroma_db_tools",
    documents=doc_splits,
    embedding=embeddings.ollama.OllamaEmbeddings(model='nomic-embed-text'),
)

vectorstore.persist()

# vdb = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embeddings.ollama.OllamaEmbeddings(
#         model='nomic-embed-text'),
# )

# print(vdb.get())


# print(vectorstore.get())

# llm = ChatOllama(model='mistral')

# retriever = vectorstore.as_retriever()

# after_rag_template = """Answer the question based only on the following context:
# {context}
# Question: {question}
# """
# after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
# after_rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | after_rag_prompt
#     | llm
#     | StrOutputParser()
# )
# print(after_rag_chain.invoke("explain langchain"))
