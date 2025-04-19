from azure.storage.blob import BlobServiceClient
import logging
import io
import PyPDF2
from rich.console import Console
from rich.progress import Progress

console = Console()

class DocumentLoader:
    def __init__(self, connection_string: str):
        """Initialize the DocumentLoader with Azure Storage connection string"""
        self.connection_string = connection_string
        if not self.connection_string:
            raise ValueError("Azure Storage connection string not found")
        
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        except Exception as e:
            console.print(f"[red]Error connecting to Azure Storage: {str(e)}[/]")
            raise

        # Configure logging
        self.logger = logging.getLogger(__name__)

    def extract_pdf_text(self, pdf_content: bytes) -> str:
        """Extract text from PDF content"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = []
            
            for page in pdf_reader.pages:
                try:
                    text.append(page.extract_text())
                except Exception as e:
                    self.logger.warning(f"Error extracting text from page: {str(e)}")
                    continue
            
            return "\n\n".join(text)
        except Exception as e:
            self.logger.error(f"Error extracting PDF text: {str(e)}")
            return ""

    def load_documents_from_container(self, container_name: str):
        """Load documents from Azure Storage container"""
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            documents = []
            
            # List all blobs in the container
            blobs = list(container_client.list_blobs())
            if not blobs:
                console.print("[yellow]No documents found in container[/]")
                return []

            with Progress() as progress:
                task = progress.add_task("[cyan]Loading documents...", total=len(blobs))
                
                for blob in blobs:
                    try:
                        if blob.name.lower().endswith('.pdf'):
                            blob_client = container_client.get_blob_client(blob.name)
                            content = blob_client.download_blob().readall()
                            text_content = self.extract_pdf_text(content)
                            
                            if text_content:
                                documents.append({
                                    'content': text_content,
                                    'source': blob.name,
                                    'metadata': {
                                        'size': blob.size,
                                        'last_modified': blob.last_modified,
                                        'type': 'pdf'
                                    }
                                })
                                console.print(f"[green]Successfully loaded:[/] {blob.name}")
                            else:
                                console.print(f"[yellow]Warning:[/] No text content extracted from {blob.name}")
                        
                        progress.update(task, advance=1)
                    
                    except Exception as e:
                        console.print(f"[red]Error loading {blob.name}: {str(e)}[/]")
                        progress.update(task, advance=1)
                        continue

            # Summary of loaded documents
            if documents:
                console.print(f"[green]Successfully loaded {len(documents)} documents[/]")
            else:
                console.print("[yellow]No valid PDF documents were loaded[/]")

            return documents

        except Exception as e:
            self.logger.error(f"Error accessing container: {str(e)}")
            console.print(f"[red]Error accessing container {container_name}: {str(e)}[/]")
            return []

    def get_container_info(self, container_name: str) -> dict:
        """Get information about the container"""
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            blobs = list(container_client.list_blobs())
            
            info = {
                'total_blobs': len(blobs),
                'pdf_count': sum(1 for blob in blobs if blob.name.lower().endswith('.pdf')),
                'total_size': sum(blob.size for blob in blobs),
                'last_modified': max((blob.last_modified for blob in blobs), default=None)
            }
            
            return info
        except Exception as e:
            self.logger.error(f"Error getting container info: {str(e)}")
            return {}

    def validate_pdf(self, pdf_content: bytes) -> bool:
        """Validate if the PDF content is readable"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            PyPDF2.PdfReader(pdf_file)
            return True
        except Exception:
            return False