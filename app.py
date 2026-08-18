import os
import certifi
import requests
import streamlit as st
from dotenv import load_dotenv

from langchain_openrouter import ChatOpenRouter
from langchain_tavily import TavilySearch
from langchain_classic.agents import create_react_agent, AgentExecutor
from langsmith import Client
from langchain.tools import tool

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

st.set_page_config(page_title="Agentic AI Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Agentic AI Assistant")
st.markdown("Search + Weather AI Agent using LangChain")

search_tool = TavilySearch(max_results=2)


@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}℃\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )


llm = ChatOpenRouter(
    model="meta-llama/llama-3.2-3b-instruct", temperature=0, api_key=OPEN_ROUTER_API_KEY
)

client = Client()

prompt = client.pull_prompt("hwchase17/react", dangerously_pull_public_prompt=True)

tools = [search_tool, get_weather_data]

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

user_query = st.text_input(
    "Enter your query:",
    placeholder="Example: Find the capital of India and current weather",
)

# ==========================================
# RUN AGENT
# ==========================================

if st.button("Run Agent"):

    if user_query:

        with st.spinner("Agent is thinking..."):

            try:
                response = agent_executor.invoke({"input": user_query})

                st.success("Response Generated")

                st.markdown("## Final Response")
                st.write(response["output"])

            except Exception as e:
                st.error(f"Error: {str(e)}")

    else:
        st.warning("Please enter a query")
