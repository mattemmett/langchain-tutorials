from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents import ZeroShotAgent
from tools.currency_tools import exchange_rate_tool

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_api_version = os.getenv("OPENAI_API_VERSION")
azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

# Define tools for the agent
tools = [
    Tool(
        name="Exchange Rate Tool",
        func=exchange_rate_tool,
        description="Fetch currency exchange rates."
    ),
]

llm = AzureChatOpenAI(
    openai_api_key=openai_api_key,
    azure_endpoint=openai_api_base,
    azure_deployment=azure_deployment_name,
    openai_api_type="azure",
    openai_api_version="2023-05-15",
)

# Initialize the agent with the correct AgentType
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Test the agent
query = "What is the current exchange rate between USD and EUR?"
try:
    response = agent.run(query)
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")