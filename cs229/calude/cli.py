import os
import sys
import argparse
import tempfile
import threading
import time
import uuid
import re
from datetime import datetime
from tqdm import tqdm
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import fitz  # PyMuPDF
import cv2
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from pymongo import MongoClient
from dotenv import load_dotenv
import readline  # For better CLI input experience

# Load environment variables
load_dotenv()

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class ProgressSpinner:
    """A simple spinner for showing progress in CLI"""
    def __init__(self, message="Processing"):
        self.message = message
        self.running = True
        self.spinner_thread = threading.Thread(target=self._spin)
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        
    def start(self):
        self.running = True
        self.spinner_thread.start()
        
    def stop(self):
        self.running = False
        if self.spinner_thread.is_alive():
            self.spinner_thread.join()
        # Clear the line
        sys.stdout.write('\r' + ' ' * (len(self.message) + 10) + '\r')
        sys.stdout.flush()
        
    def _spin(self):
        i = 0
        while self.running:
            sys.stdout.write(f'\r{self.spinner_chars[i % len(self.spinner_chars)]} {self.message}')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1

class RAGEngine:
    def __init__(self):
        """Initialize the RAG engine"""
        self.vectorstore = None
        self.conversation = None
        self.chat_history = []
        self.mongodb_client = None
        self.db = None
        self.collection = None
        self.pdf_metadata = {}
        self.current_pdf_path = None
        self.context_directory = "context"
        
        # Ensure context directory exists
        if not os.path.exists(self.context_directory):
            os.makedirs(self.context_directory)
            print(f"{Colors.GREEN}Created context directory: {self.context_directory}{Colors.ENDC}")
    
    def connect_to_mongodb(self, connection_string="mongodb://localhost:27017/"):
        """Connect to MongoDB"""
        try:
            print(f"{Colors.CYAN}Connecting to MongoDB...{Colors.ENDC}")
            self.mongodb_client = MongoClient(connection_string)
            self.db = self.mongodb_client.rag_database
            self.collection = self.db.document_chunks
            print(f"{Colors.GREEN}Connected to MongoDB successfully{Colors.ENDC}")
            return True
        except Exception as e:
            print(f"{Colors.FAIL}Error connecting to MongoDB: {e}{Colors.ENDC}")
            return False

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file with page numbers"""
        try:
            print(f"\n{Colors.CYAN}Extracting text from PDF: {os.path.basename(pdf_path)}{Colors.ENDC}")
            self.current_pdf_path = pdf_path
            pdf_reader = PdfReader(pdf_path)
            
            # Store basic PDF metadata
            self.pdf_metadata = {
                "filename": os.path.basename(pdf_path),
                "path": pdf_path,
                "total_pages": len(pdf_reader.pages),
                "processed_date": datetime.now().isoformat(),
                "table_of_contents": {}
            }
            
            pages_text = {}
            
            for i, page in enumerate(tqdm(pdf_reader.pages, desc="Extracting text")):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    page_num = i + 1
                    pages_text[page_num] = page_text
                    print(f"{Colors.GREEN}✓ Extracted text from page {page_num}{Colors.ENDC}")
                else:
                    print(f"{Colors.WARNING}⚠ No text found on page {i+1}{Colors.ENDC}")
            
            return pages_text
        except Exception as e:
            print(f"{Colors.FAIL}Error extracting text from PDF: {e}{Colors.ENDC}")
            return {}

    def analyze_page_topics(self, pages_text):
        """Use LLM to analyze topic for each page"""
        print(f"\n{Colors.CYAN}Analyzing page topics with LLM...{Colors.ENDC}")
        
        # Verify API key is available
        if not os.getenv("GROQ_API_KEY"):
            print(f"{Colors.FAIL}Groq API key not found. Please set it in .env file{Colors.ENDC}")
            return {}
        
        try:
            llm = ChatGroq(
                model_name="llama3-70b-8192",
                temperature=0.1,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            
            page_topics = {}
            toc = {}  # Table of contents
            
            spinner = ProgressSpinner("Analyzing page topics with LLM")
            spinner.start()
            
            for page_num, text in pages_text.items():
                # Create a prompt to analyze the page content
                prompt = f"""
                Analyze the following text from page {page_num} of a document and extract:
                1. The main topic(s) discussed on this page (max 2-3 topics)
                2. A brief summary of the content (2-3 sentences)
                
                Here's the text:
                {text[:1500]}...
                
                Respond in this JSON-like format only:
                TOPICS: [comma-separated list of topics]
                SUMMARY: brief summary
                """
                
                # Get LLM response
                response = llm.invoke(prompt).content
                
                # Extract topics and summary using regex
                topics_match = re.search(r"TOPICS:\s*\[(.*?)\]", response, re.DOTALL)
                summary_match = re.search(r"SUMMARY:\s*(.*?)(?:\n|$)", response, re.DOTALL)
                
                topics = []
                summary = ""
                
                if topics_match:
                    topics_str = topics_match.group(1).strip()
                    topics = [t.strip().strip('"\'') for t in topics_str.split(",")]
                
                if summary_match:
                    summary = summary_match.group(1).strip()
                
                # Store the analysis
                page_topics[page_num] = {
                    "topics": topics,
                    "summary": summary,
                    "text": text
                }
                
                # Update the table of contents
                for topic in topics:
                    if topic not in toc:
                        toc[topic] = []
                    toc[topic].append(page_num)
            
            spinner.stop()
            
            # Sort the table of contents by page number
            for topic in toc:
                toc[topic] = sorted(toc[topic])
            
            # Update PDF metadata
            self.pdf_metadata["table_of_contents"] = toc
            
            print(f"{Colors.GREEN}✓ Completed topic analysis for {len(page_topics)} pages{Colors.ENDC}")
            print(f"{Colors.GREEN}✓ Generated table of contents with {len(toc)} topics{Colors.ENDC}")
            
            return page_topics
        
        except Exception as e:
            print(f"{Colors.FAIL}Error analyzing page topics: {e}{Colors.ENDC}")
            return {}

    def capture_page_images(self, pdf_path):
        """Capture images from each page of the PDF"""
        print(f"\n{Colors.CYAN}Capturing page images...{Colors.ENDC}")
        
        try:
            doc = fitz.open(pdf_path)
            page_images = {}
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Scale factor of 2 for better quality
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                
                # Convert to RGB if needed
                if pix.n == 4:  # RGBA
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2RGB)
                
                page_images[page_num + 1] = img_data
                
            print(f"{Colors.GREEN}✓ Captured images from {len(page_images)} pages{Colors.ENDC}")
            return page_images
            
        except Exception as e:
            print(f"{Colors.FAIL}Error capturing page images: {e}{Colors.ENDC}")
            return {}

    def save_context_image(self, img_data, question_id, page_num, area=None):
        """Save image of a specific page or area as context for a question"""
        # Create directory for this question if it doesn't exist
        question_dir = os.path.join(self.context_directory, question_id)
        if not os.path.exists(question_dir):
            os.makedirs(question_dir)
        
        # Convert numpy array to PIL Image
        img = Image.fromarray(img_data)
        
        # If a specific area is provided, crop the image
        if area:
            img = img.crop((area['x1'], area['y1'], area['x2'], area['y2']))
        
        # Save the image
        img_path = os.path.join(question_dir, f"page_{page_num}.png")
        img.save(img_path)
        return img_path

    def process_documents(self, pdf_path):
        """Process PDF document for RAG"""
        # Extract text with page numbers
        pages_text = self.extract_text_from_pdf(pdf_path)
        if not pages_text:
            return False
        
        # Analyze topics for each page
        page_topics = self.analyze_page_topics(pages_text)
        if not page_topics:
            return False
        
        # Capture page images (for later use)
        page_images = self.capture_page_images(pdf_path)
        
        # Generate chunks with page number tags
        print(f"\n{Colors.CYAN}Generating text chunks with page number tags...{Colors.ENDC}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        all_chunks = []
        chunk_pages = []  # To track which page each chunk came from
        
        for page_num, page_data in page_topics.items():
            # Add page number and topic tags to the text
            tagged_text = f"[PAGE {page_num}] [TOPICS: {', '.join(page_data['topics'])}]\n{page_data['text']}"
            chunks = text_splitter.split_text(tagged_text)
            
            print(f"{Colors.GREEN}✓ Created {len(chunks)} chunks from page {page_num}{Colors.ENDC}")
            
            all_chunks.extend(chunks)
            chunk_pages.extend([page_num] * len(chunks))
        
        # Create vector store
        print(f"\n{Colors.CYAN}Creating vector embeddings... This might take a while{Colors.ENDC}")
        try:
            spinner = ProgressSpinner("Creating vector embeddings")
            spinner.start()
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            self.vectorstore = FAISS.from_texts(all_chunks, embeddings)
            
            spinner.stop()
            print(f"{Colors.GREEN}✓ Vector store created successfully with {len(all_chunks)} chunks{Colors.ENDC}")
            
            # Store in MongoDB
            if self.collection:
                print(f"\n{Colors.CYAN}Storing document chunks in MongoDB...{Colors.ENDC}")
                
                # First, store the PDF metadata
                pdf_id = str(uuid.uuid4())
                metadata_doc = {
                    "_id": pdf_id,
                    "metadata": self.pdf_metadata
                }
                self.db.pdf_metadata.insert_one(metadata_doc)
                
                # Now store the chunks with references to this PDF
                chunk_docs = []
                for i, chunk in enumerate(all_chunks):
                    chunk_docs.append({
                        "pdf_id": pdf_id,
                        "chunk_id": i,
                        "page_num": chunk_pages[i],
                        "text": chunk,
                        "topics": page_topics[chunk_pages[i]]["topics"] if chunk_pages[i] in page_topics else [],
                        "vector": embeddings.embed_query(chunk).tolist(),  # Store vector
                    })
                
                if chunk_docs:
                    self.collection.insert_many(chunk_docs)
                    print(f"{Colors.GREEN}✓ Stored {len(chunk_docs)} chunks in MongoDB{Colors.ENDC}")
            
            # Initialize conversation
            self.initialize_conversation()
            
            # Cache page images for later use
            self.page_images = page_images
            
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL}Error creating vector store: {e}{Colors.ENDC}")
            return False

    def initialize_conversation(self):
        """Initialize the conversation chain"""
        # Verify API key is available
        if not os.getenv("GROQ_API_KEY"):
            print(f"{Colors.FAIL}Groq API key not found. Please set it in .env file{Colors.ENDC}")
            return False
        
        try:
            print(f"\n{Colors.CYAN}Initializing conversation chain...{Colors.ENDC}")
            llm = ChatGroq(
                model_name="llama3-70b-8192",
                temperature=0.2,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )

            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

            self.conversation = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
                memory=memory,
                verbose=True
            )
            
            print(f"{Colors.GREEN}✓ Conversation initialized{Colors.ENDC}")
            return True
        except Exception as e:
            print(f"{Colors.FAIL}Error initializing conversation: {e}{Colors.ENDC}")
            return False

    def get_topics_from_query(self, query):
        """Extract relevant topics from user query"""
        if not os.getenv("GROQ_API_KEY"):
            print(f"{Colors.FAIL}Groq API key not found. Please set it in .env file{Colors.ENDC}")
            return []
            
        try:
            llm = ChatGroq(
                model_name="llama3-70b-8192",
                temperature=0.1,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            
            prompt = f"""
            Given the following query about a document, identify the key topics that would be relevant to search for.
            Return ONLY a comma-separated list of topics, no other text.
            Query: {query}
            """
            
            response = llm.invoke(prompt).content
            topics = [t.strip() for t in response.split(',')]
            
            print(f"{Colors.CYAN}Extracted topics from query: {', '.join(topics)}{Colors.ENDC}")
            return topics
            
        except Exception as e:
            print(f"{Colors.FAIL}Error extracting topics from query: {e}{Colors.ENDC}")
            return []

    def find_relevant_pages(self, topics):
        """Find relevant pages based on topics"""
        if not self.pdf_metadata or "table_of_contents" not in self.pdf_metadata:
            print(f"{Colors.WARNING}No table of contents available{Colors.ENDC}")
            return []
            
        toc = self.pdf_metadata["table_of_contents"]
        relevant_pages = set()
        
        for topic in topics:
            # Look for exact matches first
            if topic in toc:
                relevant_pages.update(toc[topic])
                continue
                
            # Then look for partial matches
            for toc_topic in toc:
                if topic.lower() in toc_topic.lower() or toc_topic.lower() in topic.lower():
                    relevant_pages.update(toc[toc_topic])
        
        relevant_pages = sorted(list(relevant_pages))
        if relevant_pages:
            print(f"{Colors.GREEN}Found {len(relevant_pages)} relevant pages: {relevant_pages}{Colors.ENDC}")
        else:
            print(f"{Colors.WARNING}No relevant pages found for topics. Will use general retrieval.{Colors.ENDC}")
            
        return relevant_pages

    def analyze_image_with_llm(self, image_path, question):
        """Use vision LLM to analyze an image"""
        if not os.getenv("GROQ_API_KEY"):
            print(f"{Colors.FAIL}Groq API key not found. Please set it in .env file{Colors.ENDC}")
            return ""
            
        try:
            llm = ChatGroq(
                model_name="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.1,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            
            # We need to encode the image for the LLM
            # For this implementation, we'll extract text from the image using OCR
            # and send it to the LLM along with the question
            
            try:
                img = Image.open(image_path)
                image_text = pytesseract.image_to_string(img)
            except:
                image_text = "Failed to extract text from image."
            
            prompt = f"""
            Question: {question}
            
            The following text was extracted from an image that might be relevant:
            {image_text}
            
            What information from this image would be helpful to answer the question?
            """
            
            response = llm.invoke(prompt).content
            return response
            
        except Exception as e:
            print(f"{Colors.FAIL}Error analyzing image: {e}{Colors.ENDC}")
            return ""

    def answer_query(self, query):
        """Answer a user query with references"""
        if not self.conversation:
            print(f"{Colors.FAIL}Conversation not initialized. Please process a document first.{Colors.ENDC}")
            return
        
        # Create unique ID for this question (for context images)
        question_id = str(uuid.uuid4())[:8]
        print(f"\n{Colors.BLUE}Processing query [ID: {question_id}]: {query}{Colors.ENDC}")
        
        # Extract topics from query
        query_topics = self.get_topics_from_query(query)
        
        # Find relevant pages
        relevant_pages = self.find_relevant_pages(query_topics)
        
        # Retrieve context and answer
        try:
            print(f"\n{Colors.CYAN}Retrieving relevant context...{Colors.ENDC}")
            spinner = ProgressSpinner("Generating answer")
            spinner.start()
            
            # Get answer from the conversation chain
            response = self.conversation.run(query)
            
            spinner.stop()
            
            # Extract page references from the response
            page_refs = set()
            for match in re.finditer(r'\[PAGE (\d+)\]', response):
                page_refs.add(int(match.group(1)))
            
            # Also look at context
            if hasattr(self.conversation, '_chain') and hasattr(self.conversation._chain, '_qa_chain'):
                if hasattr(self.conversation._chain._qa_chain, 'retriever'):
                    docs = self.conversation._chain._qa_chain.retriever.get_relevant_documents(query)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    print(f"\n{Colors.CYAN}Retrieved context:{Colors.ENDC}")
                    print(f"{Colors.WARNING}{context}{Colors.ENDC}")
                    
                    # Extract page references from context
                    for match in re.finditer(r'\[PAGE (\d+)\]', context):
                        page_refs.add(int(match.group(1)))
            
            # Save context images
            if page_refs and hasattr(self, 'page_images') and self.page_images:
                print(f"\n{Colors.CYAN}Saving context images...{Colors.ENDC}")
                for page_num in page_refs:
                    if page_num in self.page_images:
                        img_path = self.save_context_image(
                            self.page_images[page_num],
                            question_id,
                            page_num
                        )
                        print(f"{Colors.GREEN}✓ Saved image for page {page_num}: {img_path}{Colors.ENDC}")
            
            # Format answer with references
            pages_list = ", ".join([str(page) for page in sorted(page_refs)])
            final_answer = f"{response}\n\n{Colors.BOLD}References:{Colors.ENDC} pages {pages_list}"
            
            print(f"\n{Colors.GREEN}Answer generated with references to pages: {pages_list}{Colors.ENDC}")
            print(f"\n{Colors.GREEN}Context images saved in: {os.path.join(self.context_directory, question_id)}{Colors.ENDC}")
            
            return final_answer
            
        except Exception as e:
            print(f"{Colors.FAIL}Error generating answer: {e}{Colors.ENDC}")
            return f"Error generating answer: {e}"

def main():
    parser = argparse.ArgumentParser(description="RAG CLI for processing PDFs and answering questions")
    parser.add_argument("--pdf", type=str, help="Path to PDF file to process")
    parser.add_argument("--mongo", type=str, default="mongodb://localhost:27017/", 
                       help="MongoDB connection string (default: mongodb://localhost:27017/)")
    args = parser.parse_args()
    
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Connect to MongoDB
    rag.connect_to_mongodb(args.mongo)
    
    # If PDF is provided, process it
    if args.pdf:
        if not os.path.exists(args.pdf):
            print(f"{Colors.FAIL}Error: PDF file not found: {args.pdf}{Colors.ENDC}")
            sys.exit(1)
        
        print(f"{Colors.HEADER}Processing PDF: {args.pdf}{Colors.ENDC}")
        rag.process_documents(args.pdf)
    
    # Interactive mode
    print(f"\n{Colors.HEADER}=== RAG CLI ===={Colors.ENDC}")
    print(f"{Colors.CYAN}Enter your questions about the document (type 'exit' to quit){Colors.ENDC}")
    
    while True:
        try:
            query = input(f"\n{Colors.BOLD}Question:{Colors.ENDC} ")
            if query.lower() in ['exit', 'quit', 'q']:
                break
                
            if not query.strip():
                continue
                
            if not rag.conversation:
                print(f"{Colors.WARNING}Please process a PDF document first with --pdf option{Colors.ENDC}")
                continue
                
            answer = rag.answer_query(query)
            if answer:
                print(f"\n{Colors.BOLD}Answer:{Colors.ENDC}\n{answer}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"{Colors.FAIL}Error: {e}{Colors.ENDC}")

if __name__ == "__main__":
    main()
