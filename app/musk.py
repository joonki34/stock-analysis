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

# Define prompt
prompt_template = """
You are a professional Wall Street analyst. I want you to predict if an Elon Musk's tweet would have positive effect on the stock price of Tesla Inc (TSLA). Here's one of the Elon Musk's tweet below:
---
{tweet}
---

The report should have the following format:
---
Prediction: Positive / Negative
Reason:
---
"""
prompt = PromptTemplate.from_template(prompt_template)

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# Create a language chain with the LLM and the parser
chain = prompt | llm | parser

print(chain.invoke({"tweet": "Next I'm buying Coca-Cola to put the cocaine back in."}))

# from langchain_community.document_loaders import TwitterTweetLoader

# twitter_token = os.environ['TWITTER_TOKEN']

# loader = TwitterTweetLoader.from_bearer_token(
#     oauth2_bearer_token=twitter_token,
#     twitter_users=["elonmusk"],
#     number_tweets=10,  # Default value is 100
# )

# documents = loader.load()
# for document in documents[1:2]:
#     content = document.page_content
#     print("Content: ", content)