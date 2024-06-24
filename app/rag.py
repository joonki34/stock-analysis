from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

# Get the Google API key from the environment variables
api_key = os.environ['GOOGLE_API_KEY']

set_llm_cache(InMemoryCache())

# Create a Google Generative AI instance with the specified model and API key
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-latest", google_api_key=api_key, cache=True)

import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("What are the types of memory in LLM agents?"))

# cleanup
vectorstore.delete_collection()