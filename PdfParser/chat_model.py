#!/usr/bin/env python3
"""
NCERT PDF Chatbot
A terminal-based application that processes PDFs and allows querying their content with context awareness
"""

import os
import sys
import argparse
import logging
from dotenv import load_dotenv
from termcolor import colored
from rich.console import Console
from rich.markdown import Markdown
from typing import List, Dict, Any, Optional

# Import custom modules
from pdf_processor import PDFProcessor
from database import MongoDBHandler
from conversation_manager import ConversationManager
from chatbot import Chatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='NCERT PDF Chatbot')
    parser.add_argument('--pdf', type=str, help='Path to PDF file or directory containing PDFs')
    parser.add_argument('--mongodb_uri', type=str, help='MongoDB connection URI')
    parser.add_argument('--groq_api_key', type=str, help='Groq API Key')
    parser.add_argument('--reindex', action='store_true', help='Force reindexing of PDFs')
    return parser.parse_args()

def check_environment():
    """Check if required environment variables are set."""
    required_vars = ['MONGODB_URI', 'GROQ_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
        logger.info("Please set these in a .env file or as environment variables")
        sys.exit(1)

def get_config(args):
    """Get configuration from args and environment variables."""
    config = {
        'mongodb_uri': args.mongodb_uri or os.getenv('MONGODB_URI'),
        'groq_api_key': args.groq_api_key or os.getenv('GROQ_API_KEY'),
        'pdf_path': args.pdf,
        'reindex': args.reindex
    }
    return config

def print_welcome_message():
    """Print welcome message and instructions."""
    console = Console()
    console.print("\n[bold blue]Welcome to NCERT PDF Chatbot![/bold blue]\n")
    console.print("This chatbot allows you to query NCERT textbooks and get accurate answers.")
    console.print("The chatbot understands text and diagrams in the PDFs.")
    console.print("\n[bold green]Commands:[/bold green]")
    console.print("- Type your question and press Enter")
    console.print("- Type [bold]/add [pdf_path][/bold] to add a new PDF")
    console.print("- Type [bold]/list[/bold] to see all indexed PDFs")
    console.print("- Type [bold]/clear[/bold] to start a new conversation")
    console.print("- Type [bold]/exit[/bold] or [bold]/quit[/bold] to exit\n")

def main():
    """Main entry point for the application."""
    args = parse_arguments()
    
    # If no arguments provided, check environment variables
    if not (args.mongodb_uri or args.groq_api_key):
        check_environment()
    
    config = get_config(args)
    
    # Initialize components
    db_handler = MongoDBHandler(config['mongodb_uri'])
    pdf_processor = PDFProcessor(db_handler)
    conversation_manager = ConversationManager(db_handler)
    chatbot = Chatbot(config['groq_api_key'], db_handler, conversation_manager)
    
    # Process PDF if provided
    if config['pdf_path']:
        if os.path.isdir(config['pdf_path']):
            # Process all PDFs in directory
            pdf_files = [os.path.join(config['pdf_path'], f) for f in os.listdir(config['pdf_path']) 
                         if f.lower().endswith('.pdf')]
            for pdf_file in pdf_files:
                pdf_processor.process_pdf(pdf_file, force_reindex=config['reindex'])
        elif os.path.isfile(config['pdf_path']):
            # Process single PDF file
            pdf_processor.process_pdf(config['pdf_path'], force_reindex=config['reindex'])
        else:
            logger.error(f"Path not found: {config['pdf_path']}")
            sys.exit(1)
    
    console = Console()
    print_welcome_message()
    
    # Interactive chat loop
    conversation_id = conversation_manager.create_conversation()
    
    while True:
        try:
            user_input = input(colored("\nYou: ", "green"))
            
            # Handle special commands
            if user_input.lower() in ['/exit', '/quit']:
                print(colored("Goodbye!", "blue"))
                break
            elif user_input.lower() == '/clear':
                conversation_id = conversation_manager.create_conversation()
                print(colored("Started a new conversation.", "blue"))
                continue
            elif user_input.lower() == '/list':
                pdfs = db_handler.list_indexed_pdfs()
                console.print("\n[bold]Indexed PDFs:[/bold]")
                for i, pdf in enumerate(pdfs, 1):
                    console.print(f"{i}. {pdf['title']} ({pdf['file_path']})")
                continue
            elif user_input.lower().startswith('/add '):
                pdf_path = user_input[5:].strip()
                if os.path.isfile(pdf_path):
                    pdf_processor.process_pdf(pdf_path)
                else:
                    print(colored(f"File not found: {pdf_path}", "red"))
                continue
            
            # Process regular query
            response = chatbot.process_query(user_input, conversation_id)
            
            # Display response with markdown formatting
            console.print("\n[bold blue]Assistant:[/bold blue]")
            console.print(Markdown(response))
            
        except KeyboardInterrupt:
            print(colored("\nGoodbye!", "blue"))
            break
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            print(colored(f"An error occurred: {str(e)}", "red"))

if __name__ == "__main__":
    main()