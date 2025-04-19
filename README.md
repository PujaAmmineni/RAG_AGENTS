# Azure RAG Agents - Multi-Agent Retrieval-Augmented Generation System

This project answers user questions using a Retrieval-Augmented Generation (RAG) pipeline powered by Azure OpenAI, FAISS, and multiple collaborating AI agents. PDF documents are retrieved from Azure Blob Storage, semantically searched using embeddings, and interpreted by AutoGen-style agents to generate accurate and context-rich responses.

## Features

- Vector search using FAISS and Azure OpenAI embeddings
- Multi-agent reasoning system with the following roles:
  - Commander: Analyzes the question and context
  - Prover: Searches and provides evidence from documents
  - Verifier: Validates the response and provides a summary
  - Generalist: Handles fallback queries not answered by documents
- Confidence score (1 to 10) shown for each agent when relevant context is found
- Fallback response from Generalist agent when no answer is found in documents
- Interactive command-line interface for question answering


## Setup Instructions

### Step 1: Clone the Repository

Clone the project and navigate into the directory.

```bash
git clone https://github.com/PujaAmmineni/RAG_AGENTS.git
cd RAG_AGENTS
python -m venv ragenv
ragenv\Scripts\activate
pip install -r requirements.txt
AZURE_OPENAI_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-35-turbo
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
CHAT_API_VERSION=2024-02-15-preview
EMBEDDING_API_VERSION=2023-05-15
AZURE_STORAGE_CONNECTION_STRING=your-azure-blob-storage-connection-string
python main.py

