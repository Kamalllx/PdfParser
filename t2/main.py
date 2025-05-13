#!/usr/bin/env python3
"""
NCERT Study Assistant - A CLI chatbot for NCERT textbooks

This application loads PDF files, extracts text and images, creates vector
embeddings using Groq and stores them in Pinecone for efficient retrieval.
The chatbot can answer questions about the content, including diagrams.
"""

import os
import sys
import argparse
from typing import List, Dict, Any, Optional
import logging
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown
import dotenv

from pdf_processor import NCERTPDFProcessor
from vector_store import PineconeVectorStore
from chatbot import GroqChatbot
from conversation_manager import ConversationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize Rich console for better CLI output
console = Console()

def setup_environment() -> bool:
    """Load environment variables and check for required API keys."""
    dotenv.load_dotenv()
    
    required_keys = ["GROQ_API_KEY", "PINECONE_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        console.print(f"[bold red]Missing required API keys: {', '.join(missing_keys)}[/bold red]")
        console.print("Please add them to your .env file or set them as environment variables.")
        return False
    
    logger.info("Environment variables loaded successfully")
    return True

def create_argparser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(description="NCERT Study Assistant - A chatbot for NCERT textbooks")
    
    parser.add_argument(
        "--pdf", "-p",
        type=str,
        help="Path to a PDF file or directory containing PDFs to process"
    )
    
    parser.add_argument(
        "--index_name", "-i",
        type=str, 
        default="ncert-assistant",
        help="Name of the Pinecone index to use (default: ncert-assistant)"
    )
    
    parser.add_argument(
        "--dimension", "-d",
        type=int,
        default=1536,
        help="Dimension of the vector embeddings (default: 1536)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llama3-70b-8192",
        help="Groq model to use (default: llama3-70b-8192)"
    )
    
    parser.add_argument(
        "--vision_model", "-v",
        type=str,
        default="llava-13b",
        help="Groq vision model to use for image understanding (default: llava-13b)"
    )
    
    parser.add_argument(
        "--chunk_size", "-c",
        type=int,
        default=1000,
        help="Chunk size for text splitting (default: 1000)"
    )
    
    parser.add_argument(
        "--chunk_overlap", "-o",
        type=int,
        default=200,
        help="Chunk overlap for text splitting (default: 200)"
    )
    
    return parser

def process_pdfs(pdf_path: str, pdf_processor: NCERTPDFProcessor) -> bool:
    """Process PDF files from the given path."""
    if not os.path.exists(pdf_path):
        console.print(f"[bold red]Error: Path not found: {pdf_path}[/bold red]")
        return False
    
    if os.path.isfile(pdf_path):
        if not pdf_path.lower().endswith('.pdf'):
            console.print(f"[bold yellow]Warning: {pdf_path} is not a PDF file. Skipping.[/bold yellow]")
            return False
        
        console.print(f"[bold green]Processing PDF: {pdf_path}[/bold green]")
        try:
            pdf_processor.process_pdf(pdf_path)
            console.print(f"[bold green]Successfully processed: {pdf_path}[/bold green]")
            return True
        except Exception as e:
            console.print(f"[bold red]Error processing {pdf_path}: {str(e)}[/bold red]")
            return False
    
    elif os.path.isdir(pdf_path):
        console.print(f"[bold blue]Processing PDF directory: {pdf_path}[/bold blue]")
        success = False
        for root, _, files in os.walk(pdf_path):
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    try:
                        console.print(f"[blue]Processing PDF: {file_path}[/blue]")
                        pdf_processor.process_pdf(file_path)
                        console.print(f"[green]Successfully processed: {file_path}[/green]")
                        success = True
                    except Exception as e:
                        console.print(f"[yellow]Error processing {file_path}: {str(e)}[/yellow]")
        
        return success
    
    return False

def chat_loop(chatbot: GroqChatbot, conversation_manager: ConversationManager):
    """Run the interactive chat loop."""
    console.print(Panel.fit(
        "[bold blue]NCERT Study Assistant[/bold blue]\n"
        "[italic]Ask questions about your NCERT textbooks, including diagrams and images.[/italic]\n"
        "Type 'exit' or 'quit' to end the session."
    ))
    
    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/bold green]")
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                console.print("[bold blue]Thank you for using NCERT Study Assistant. Goodbye![/bold blue]")
                break
                
            # Process the user query
            with console.status("[bold green]Thinking...[/bold green]", spinner="dots"):
                response = chatbot.chat(user_input)
            
            # Display response with markdown formatting
            console.print("[bold purple]Assistant:[/bold purple]")
            console.print(Markdown(response))
            
        except KeyboardInterrupt:
            console.print("\n[bold blue]Session terminated. Thank you for using NCERT Study Assistant![/bold blue]")
            break
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")

def main():
    """Main entry point for the application."""
    # Check for required environment variables
    if not setup_environment():
        return 1
    
    # Parse command-line arguments
    parser = create_argparser()
    args = parser.parse_args()
    
    try:
        # Initialize the vector store
        console.print("[bold blue]Initializing vector database...[/bold blue]")
        vector_store = PineconeVectorStore(
            api_key=os.getenv("PINECONE_API_KEY"),
            index_name=args.index_name,
            dimension=args.dimension
        )
        
        # Initialize the PDF processor
        console.print("[bold blue]Initializing PDF processor...[/bold blue]")
        pdf_processor = NCERTPDFProcessor(
            vector_store=vector_store,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            vision_model=args.vision_model
        )
        
        # Initialize the conversation manager
        conversation_manager = ConversationManager()
        
        # Initialize the chatbot
        console.print("[bold blue]Initializing chatbot...[/bold blue]")
        chatbot = GroqChatbot(
            vector_store=vector_store,
            conversation_manager=conversation_manager,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model=args.model
        )
        
        # Process PDFs if a path was provided
        if args.pdf:
            console.print("[bold blue]Processing PDFs...[/bold blue]")
            if not process_pdfs(args.pdf, pdf_processor):
                console.print("[bold yellow]Warning: No PDFs were processed successfully.[/bold yellow]")
                if not vector_store.has_documents():
                    console.print("[bold red]Error: No documents found in vector store. Please add PDFs first.[/bold red]")
                    return 1
        elif not vector_store.has_documents():
            console.print("[bold red]Error: No documents found in vector store. Please specify a PDF path with --pdf.[/bold red]")
            return 1
        
        # Start the chat loop
        chat_loop(chatbot, conversation_manager)
        
        return 0
    
    except Exception as e:
        console.print(f"[bold red]Critical error: {str(e)}[/bold red]")
        return 1

if __name__ == "__main__":
    sys.exit(main())