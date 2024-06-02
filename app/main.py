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

from langchain_core.messages import HumanMessage, SystemMessage

# Create a list of messages for the language chain
messages = [
    SystemMessage(content="아래의 내용을 두 문장으로 요약해주세요:"),
    HumanMessage(content="""
작가주의란 용어는 프랑스의 영화감독이자 이론가였던 프랑수아 트뤼포가 영화비평 전문잡지 『카이에 뒤 시네마』(1954, 1)에 발표한 일종의 영화이론이었다. 이 용어가 60년대 이후 번역, 소개되면서 국제적인 비평 용어로 널리 사용되기에 이르렀다.

작가주의 영화는 영화적인 컨벤션을 거부하는 데서 시작한다. 이때 작가란 시나리오 작가가 아니라 영화감독을 가리키고 있다. 즉, 작가주의 영화는 비장르 영화, 작가예술 영화, 감독의 개성과 독창성이 중시되는 영화를 뜻한다. 작가주의 영화의 특징은 다음과 같이 세 가지로 요약될 수 있다. 첫째, 독자적인 견해와 방식을 창출함으로써 관습을 변형하거나 장르를 생성한다. 둘째, 영화가 제기하는 문제들을 단순명쾌하게 해결하지 못한다. 셋째, 감독은 스튜디오의 조건을 지배한다.

작가주의를 처음으로 제창했던 트뤼포의 영화, 이를테면, 어린시절의 자전적 체험을 영화화한 「사백번의 구타」(1958), 영화의 음악적 요소를 강조한 「피아니스트를 쏴라」(1960), 세 사람 남녀 간의 비극적인 사랑을 묘파한 「줄과 짐」(1961) 등의 경우도 작가주의 영화라고 할 수 있다. 그는 형식에 얽매이지 않는 자유분방한 연출 스타일을 시도해 작가주의 이론이 누벨바그 운동으로 이어지는 교량 역할을 했다.

작가주의 영화는 낯 익는 관습보다는 실험정신을 추구하기 때문에 철저한 비대중성을 지향한다. 대신에 이 영화는 영화를 예술적 의식의 소산으로 간주하면서 진지한 주제와 자유로운 양식을 좇는다. 신뢰할만한 영화 감독이라면 개성적인 표현과 창의적인 연출에 자신의 열정을 바치지 않을 수 없다.(송희복)
                 """),
]

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# Create a language chain with the LLM and the parser
chain = llm | parser

# Invoke the language chain with the messages
print(chain.invoke(messages))