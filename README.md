
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
```

### Step 2: Create and Activate Virtual Environment

#### On Windows:

```bash
python -m venv ragenv
ragenv\Scripts\activate
```


### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in your root directory with the following content:

```
AZURE_OPENAI_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-35-turbo
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
CHAT_API_VERSION=2024-02-15-preview
EMBEDDING_API_VERSION=2023-05-15
AZURE_STORAGE_CONNECTION_STRING=your-azure-blob-storage-connection-string
```

> ⚠️ Make sure not to share or commit this file.

### Step 5: Upload PDF Files to Azure Blob Storage

1. Go to the [Azure Portal](https://portal.azure.com/)
2. Navigate to your Azure Storage Account
3. Create or open a container named `content`
4. Upload your `.pdf` documents there

Alternatively, you can use Azure Storage Explorer for easier management.

### Step 6: Run the Application

```bash
python main.py
```

### Step 7: Start Asking Questions

Once the application is running, you can start asking questions like:

```bash
Enter your question (or 'exit' to quit): What is BIM?
```

The agents will retrieve relevant context, analyze it, and generate multi-step reasoning responses.

---


