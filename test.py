from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT")
)

result = embedding.embed_query("Test query")
print(" Success! Embedding result length:", len(result))
