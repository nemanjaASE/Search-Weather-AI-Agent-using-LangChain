import os
import certifi
from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter
from langchain_tavily import TavilySearch
from langchain_classic.agents import create_react_agent, AgentExecutor
from langsmith import Client

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

search_tool = TavilySearch(max_results=2)

llm = ChatOpenRouter(
    model="meta-llama/llama-3.2-3b-instruct", temperature=0, api_key=OPEN_ROUTER_API_KEY
)

client = Client()

prompt = client.pull_prompt("hwchase17/react", dangerously_pull_public_prompt=True)

tools = [search_tool]

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

response = agent_executor.invoke(
    {
        "input": ("FInd the capital of India."),
    }
)

print(response)
