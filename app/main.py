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

from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# Define prompt
prompt_template = """Write a concise summary of the following in Korean language:
"{text}"
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# Create a language chain with the LLM and the parser
chain = prompt | llm | parser

from langchain_community.document_loaders import RedditPostsLoader

reddit_client_id = os.environ['REDDIT_CLIENT_ID']
reddit_client_secret = os.environ['REDDIT_CLIENT_SECRET']

# load using 'subreddit' mode
loader = RedditPostsLoader(
    client_id=reddit_client_id,
    client_secret=reddit_client_secret,
    user_agent="extractor by u/Master_Ocelot8179",
    categories=["hot"],  # List of categories to load posts from
    mode="subreddit",
    search_queries=[
        "investing",
    ],  # List of subreddits to load posts from
    number_posts=10,  # Default value is 10
)

documents = loader.load()
for document in documents[1:2]:
    content = document.page_content
    print("Content: ", content)
    print("\n")
    print("Summary: ", chain.invoke({"text": content}))