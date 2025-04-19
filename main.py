import logging
from typing import Dict, List
import warnings
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from config import (
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    CHAT_API_VERSION,
    EMBEDDING_API_VERSION,
    AZURE_STORAGE_CONNECTION_STRING
)
from utils.document_loader import DocumentLoader
from agents import RAGAgents

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)
console = Console()

class RAGApplication:
    def __init__(self):
        self.setup_components()

    def setup_components(self):
        try:
            self.azure_config = {
                "api_key": AZURE_OPENAI_KEY,
                "api_version": CHAT_API_VERSION,
                "azure_endpoint": AZURE_OPENAI_ENDPOINT.rstrip('/'),
                "chat_deployment": AZURE_OPENAI_CHAT_DEPLOYMENT,
                "embedding_deployment": AZURE_OPENAI_EMBEDDING_DEPLOYMENT
            }

            console.print(f"[yellow]Initializing with configurations:[/]")
            for key, value in self.azure_config.items():
                if 'key' not in key.lower():
                    console.print(f"  {key}: {value}")

            self.setup_llm_and_embeddings()
            self.agents = RAGAgents(self.azure_config)
            self.doc_loader = DocumentLoader(AZURE_STORAGE_CONNECTION_STRING)
            console.print("[green]✓[/] Components initialized successfully")

        except Exception as e:
            logger.error(f"Setup error: {str(e)}", exc_info=True)
            raise

    def setup_llm_and_embeddings(self):
        try:
            self.llm = AzureChatOpenAI(
                azure_deployment=self.azure_config["chat_deployment"],
                openai_api_version=self.azure_config["api_version"],
                azure_endpoint=self.azure_config["azure_endpoint"],
                api_key=self.azure_config["api_key"],
                temperature=0.7
            )

            self.embeddings = AzureOpenAIEmbeddings(
                deployment=self.azure_config["embedding_deployment"],
                model="text-embedding-ada-002",
                api_key=self.azure_config["api_key"],
                api_version=EMBEDDING_API_VERSION,
                azure_endpoint=self.azure_config["azure_endpoint"]
            )
        except Exception as e:
            logger.error(f"Error setting up LLM and embeddings: {str(e)}", exc_info=True)
            raise

    def create_vector_store(self, documents: List[Dict]) -> FAISS:
        try:
            with Progress() as progress:
                task = progress.add_task("[cyan]Processing documents...", total=len(documents))
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

                texts, metadata = [], []
                for doc in documents:
                    chunks = text_splitter.split_text(doc['content'])
                    texts.extend(chunks)
                    metadata.extend([{"source": doc['source']} for _ in chunks])
                    progress.update(task, advance=1)

                console.print(f"[green]Created {len(texts)} text chunks from {len(documents)} documents[/]")
                return FAISS.from_texts(texts, self.embeddings, metadatas=metadata)

        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}", exc_info=True)
            raise

    def process_query(self, query: str, vectorstore: FAISS) -> Dict:
        try:
            relevant_docs = vectorstore.similarity_search(query, k=3)
            context_parts = []
            sources = set()

            for doc in relevant_docs:
                source = doc.metadata['source']
                content = doc.page_content
                context_parts.append(f"[From {source}]:\n{content}")
                sources.add(source)

            context = "\n\n".join(context_parts)
            chat_response = self.agents.process_query(query, context)

            return {
                "question": query,
                "context": context,
                "chat_response": chat_response,
                "sources": list(sources)
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {"error": str(e)}

    def handle_agent_response(self, response: Dict) -> None:
        try:
            if "error" in response:
                console.print(Panel(f"[red]Error: {response['error']}[/]"))
                return

            console.print(Panel(
                f"[bold]Question:[/] {response['question']}\n\n"
                f"[bold]Using context from:[/] {', '.join(response['sources']) if response['sources'] else '❌ No context from documents'}",
                title="Query Information",
                border_style="blue"
            ))

            if response.get("chat_response"):
                self.display_agent_outcome(response["chat_response"])

        except Exception as e:
            logger.error(f"Error handling response: {str(e)}", exc_info=True)
            console.print("[red]Error displaying response[/]")

    def display_agent_outcome(self, chat_response: str) -> None:
        try:
            if chat_response.startswith("GENERAL ANSWER:"):
                console.print(Panel(chat_response, title="🧠 Generalist Agent Response", border_style="magenta"))
                return

            scores = {
                "Commander": 9,
                "Prover": 8,
                "Verifier": 9
            }

            parts = {
                "ANALYSIS:": ("Commander Analysis", "blue", "Commander"),
                "EVIDENCE:": ("Evidence", "green", "Prover"),
                "FINAL ANSWER:": ("Final Summary", "cyan", "Verifier")
            }

            upper_response = chat_response.upper()
            for marker, (title, color, agent) in parts.items():
                marker_upper = marker.upper()
                if marker_upper in upper_response:
                    start = upper_response.find(marker_upper)
                    end = min(
                        [upper_response.find(m, start) for m in parts if m != marker and upper_response.find(m, start) != -1] + [len(chat_response)]
                    )
                    content = chat_response[start:end].strip()
                    console.print(Panel(f"{content}\n\nConfidence Score: {scores[agent]}/10", title=title, border_style=color))

        except Exception as e:
            console.print(Panel(str(e), title="Error Displaying Agent Response", border_style="red"))


def main():
    console.print(Panel.fit(
        "[bold blue]Azure OpenAI RAG Application with Agents[/]",
        subtitle="[italic]Powered by Azure OpenAI & FAISS[/]"
    ))

    try:
        app = RAGApplication()
        with console.status("[yellow]Loading documents from Azure Storage...[/]"):
            documents = app.doc_loader.load_documents_from_container("content")

        if not documents:
            console.print("[red]No documents found in storage container![/]")
            return

        console.print("\n[yellow]Creating vector store...[/]")
        vectorstore = app.create_vector_store(documents)
        console.print("[green]✓[/] Vector store created successfully")

        while True:
            query = console.input("\n[bold yellow]Enter your question (or 'exit' to quit):[/] ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            if not query or query == "?":
                console.print("[yellow]Please enter a valid question.[/]")
                continue

            with console.status("[bold blue]Processing your query...[/]"):
                response = app.process_query(query, vectorstore)
                app.handle_agent_response(response)

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        console.print(f"[red]Application error: {str(e)}[/]")
        return

    console.print("\n[green]Thank you for using the RAG Application![/]")

if __name__ == "__main__":
    main()