from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_api_version = os.getenv("OPENAI_API_VERSION")
azure_deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")



model = AzureChatOpenAI(
    openai_api_key=openai_api_key,
    azure_endpoint=openai_api_base,
    azure_deployment=azure_deployment_name,
    openai_api_type="azure",
    openai_api_version="2023-05-15",
)

messages = [
        SystemMessage(content="Translate the following from English into Italian"),
        HumanMessage(content="hi!"),
    ]

response = model.invoke(messages)

print(response)