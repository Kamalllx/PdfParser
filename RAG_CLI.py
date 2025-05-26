import os
import argparse
import tempfile
from PyPDF2 import PdfReader
import fitz  # PyMuPDF for screenshots
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pymongo
from datetime import datetime
import json
import re
from typing import List, Dict, Any
import shutil

# Load environment variables
load_dotenv()

class RAGCLIApp:
    def __init__(self):
        print("🚀 Initializing RAG CLI Application...")
        
        # Initialize embeddings
        print("🔧 Setting up embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        print("✅ Embeddings model loaded")
        
        # Initialize LLMs
        print("🔧 Setting up Groq LLMs...")
        self.llm = ChatGroq(
            model_name="llama3-70b-8192",
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        print("✅ Main LLM initialized")
        
        self.vision_llm = ChatGroq(
            model_name="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        print("✅ Vision LLM initialized")
        
        # MongoDB setup
        print("🔧 Connecting to MongoDB...")
        try:
            mongo_client = pymongo.MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
            self.db = mongo_client["rag_database"]
            self.collection = self.db["documents"]
            # Test connection
            self.db.list_collection_names()
            print("✅ Connected to MongoDB successfully")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            raise
        
        print("🎉 RAG CLI App initialized successfully!")
        
    def extract_pages_with_analysis(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF pages and analyze topics"""
        print(f"\n📖 Starting PDF processing: {os.path.basename(pdf_path)}")
        
        try:
            print("🔄 Opening PDF file...")
            pdf_reader = PdfReader(pdf_path)
            doc = fitz.open(pdf_path)  # For screenshots
            
            pages_data = []
            total_pages = len(pdf_reader.pages)
            
            print(f"📄 Total pages found: {total_pages}")
            
            for page_num, page in enumerate(pdf_reader.pages, 1):
                print(f"\n🔄 Processing page {page_num}/{total_pages}")
                
                # Extract text
                print(f"📝 Extracting text from page {page_num}...")
                text = page.extract_text() or ""
                
                if not text.strip():
                    print(f"⚠️  Page {page_num} contains no text, skipping...")
                    continue
                
                # Tag content with page number
                tagged_content = f"[PAGE_{page_num}] {text}"
                
                print(f"✅ Extracted text from page {page_num} ({len(text)} characters)")
                
                # Analyze topics using LLM
                print(f"🧠 Analyzing topics for page {page_num}...")
                topic_analysis = self.analyze_page_topics(text, page_num)
                
                page_data = {
                    "page_number": page_num,
                    "raw_content": text,
                    "tagged_content": tagged_content,
                    "topics": topic_analysis["topics"],
                    "summary": topic_analysis["summary"],
                    "keywords": topic_analysis["keywords"]
                }
                
                pages_data.append(page_data)
                print(f"✅ Page {page_num} analysis complete")
                print(f"📝 Topics found: {', '.join(topic_analysis['topics'])}")
                print(f"📋 Summary: {topic_analysis['summary']}")
            
            doc.close()
            print(f"\n🎉 PDF processing complete! Successfully processed {len(pages_data)} pages")
            return pages_data
            
        except Exception as e:
            print(f"❌ Error processing PDF: {e}")
            return []
    
    def analyze_page_topics(self, text: str, page_num: int) -> Dict[str, Any]:
        """Analyze topics and content of a page using LLM"""
        print(f"🔍 Running LLM analysis for page {page_num}...")
        
        prompt = f"""
        Analyze the following text from page {page_num} and provide:
        1. Main topics/themes (list of 3-5 key topics)
        2. Brief summary (1-2 sentences)
        3. Important keywords (5-10 keywords)
        
        Text:
        {text[:2000]}  # Limit text to avoid token limits
        
        Respond in this exact JSON format:
        {{
            "topics": ["topic1", "topic2", "topic3"],
            "summary": "Brief summary here",
            "keywords": ["keyword1", "keyword2", "keyword3"]
        }}
        """
        
        try:
            print("🤖 Sending request to LLM...")
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            print("✅ LLM response received")
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis = json.loads(json_str)
                print("✅ JSON parsed successfully")
                return analysis
            else:
                # Fallback if JSON parsing fails
                print("⚠️  JSON parsing failed, using fallback")
                return {
                    "topics": ["General Content"],
                    "summary": f"Content from page {page_num}",
                    "keywords": ["content", "page", str(page_num)]
                }
                
        except Exception as e:
            print(f"⚠️  Error analyzing page {page_num}: {e}")
            return {
                "topics": ["General Content"],
                "summary": f"Content from page {page_num}",
                "keywords": ["content", "page", str(page_num)]
            }
    
    def create_table_of_contents(self, pages_data: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Create table of contents with topics and page numbers"""
        print("\n📚 Creating table of contents...")
        
        toc = {}
        
        print("🔄 Processing topics from all pages...")
        for page_data in pages_data:
            page_num = page_data["page_number"]
            topics = page_data["topics"]
            
            for topic in topics:
                topic_clean = topic.strip().lower()
                if topic_clean not in toc:
                    toc[topic_clean] = []
                    print(f"📖 New topic discovered: {topic_clean}")
                if page_num not in toc[topic_clean]:
                    toc[topic_clean].append(page_num)
        
        # Sort page numbers for each topic
        print("🔄 Sorting page numbers for each topic...")
        for topic in toc:
            toc[topic].sort()
        
        print(f"✅ Table of contents created with {len(toc)} unique topics")
        print("\n📋 TABLE OF CONTENTS:")
        print("=" * 50)
        for topic, pages in toc.items():
            print(f"📖 {topic.title()}: Pages {', '.join(map(str, pages))}")
        print("=" * 50)
        
        return toc
    
    def vectorize_content(self, pages_data: List[Dict[str, Any]]) -> FAISS:
        """Vectorize tagged content"""
        print("\n🔢 Starting vectorization process...")
        
        texts = []
        metadatas = []
        
        print("🔄 Preparing content for vectorization...")
        for page_data in pages_data:
            # Use tagged content for vectorization
            texts.append(page_data["tagged_content"])
            metadatas.append({
                "page_number": page_data["page_number"],
                "topics": page_data["topics"],
                "summary": page_data["summary"],
                "keywords": page_data["keywords"]
            })
            print(f"✅ Prepared page {page_data['page_number']} for vectorization")
        
        print("🔄 Creating FAISS vector store...")
        print("⏳ This may take a moment for large documents...")
        
        try:
            vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            print(f"🎉 Vectorization complete! Created vectors for {len(texts)} pages")
            return vectorstore
        except Exception as e:
            print(f"❌ Error during vectorization: {e}")
            raise
    
    def store_in_mongodb(self, filename: str, pages_data: List[Dict[str, Any]], 
                        toc: Dict[str, List[int]], vectorstore: FAISS):
        """Store all data in MongoDB"""
        print(f"\n💾 Storing data in MongoDB for document: {filename}")
        
        print("🔄 Serializing vector store...")
        # Serialize vectorstore
        temp_dir = tempfile.mkdtemp()
        vectorstore_path = os.path.join(temp_dir, "vectorstore")
        vectorstore.save_local(vectorstore_path)
        
        # Read serialized files
        print("🔄 Reading serialized vector data...")
        with open(os.path.join(vectorstore_path, "index.faiss"), "rb") as f:
            faiss_index = f.read()
        with open(os.path.join(vectorstore_path, "index.pkl"), "rb") as f:
            faiss_metadata = f.read()
        
        print("🔄 Preparing document data...")
        document_data = {
            "filename": filename,
            "upload_date": datetime.now(),
            "pages_data": pages_data,
            "table_of_contents": toc,
            "total_pages": len(pages_data),
            "faiss_index": faiss_index,
            "faiss_metadata": faiss_metadata
        }
        
        # Store or update document
        print("🔄 Checking for existing document...")
        existing = self.collection.find_one({"filename": filename})
        if existing:
            print("🔄 Updating existing document...")
            self.collection.update_one(
                {"filename": filename},
                {"$set": document_data}
            )
            print(f"🔄 Updated existing document: {filename}")
        else:
            print("🔄 Storing new document...")
            self.collection.insert_one(document_data)
            print(f"✅ Stored new document: {filename}")
        
        # Cleanup
        print("🔄 Cleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print("✅ Data successfully stored in MongoDB")
        print(f"📊 Document statistics:")
        print(f"   - Total pages: {len(pages_data)}")
        print(f"   - Total topics: {len(toc)}")
        print(f"   - Storage date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def find_relevant_topics(self, question: str, toc: Dict[str, List[int]]) -> List[str]:
        """Find topics relevant to the question"""
        print(f"\n🔍 Searching for topics relevant to: '{question}'")
        
        question_lower = question.lower()
        relevant_topics = []
        
        print("🔄 Performing keyword-based topic matching...")
        # Simple keyword matching
        for topic, pages in toc.items():
            if any(word in topic for word in question_lower.split() if len(word) > 2):
                relevant_topics.append(topic)
                print(f"✅ Keyword match found: {topic}")
        
        # If no direct match, use LLM to find relevant topics
        if not relevant_topics:
            print("🧠 No keyword matches found, using LLM for topic matching...")
            topic_list = list(toc.keys())
            prompt = f"""
            Given this question: "{question}"
            And these available topics: {', '.join(topic_list)}
            
            Which topics are most relevant to answer this question?
            Return only the topic names that are relevant, separated by commas.
            If no topics are directly relevant, return the 3 most potentially useful ones.
            """
            
            try:
                print("🤖 Sending topic matching request to LLM...")
                response = self.llm.invoke(prompt)
                suggested_topics = [t.strip().lower() for t in response.content.split(',')]
                relevant_topics = [t for t in suggested_topics if t in toc]
                print(f"✅ LLM suggested topics: {', '.join(relevant_topics)}")
            except Exception as e:
                print(f"⚠️  Error in LLM topic matching: {e}")
                relevant_topics = list(toc.keys())[:3]  # Fallback to first 3 topics
                print(f"🔄 Using fallback topics: {', '.join(relevant_topics)}")
        
        if not relevant_topics:
            relevant_topics = list(toc.keys())[:5]  # Final fallback
            print(f"🔄 Using top 5 topics as final fallback: {', '.join(relevant_topics)}")
        
        print(f"✅ Final relevant topics selected: {', '.join(relevant_topics)}")
        return relevant_topics
    
    def retrieve_pages_for_topics(self, topics: List[str], toc: Dict[str, List[int]]) -> List[int]:
        """Get ordered list of page numbers for topics"""
        print(f"\n📄 Retrieving pages for topics: {', '.join(topics)}")
        
        page_numbers = set()
        for topic in topics:
            if topic in toc:
                topic_pages = toc[topic]
                page_numbers.update(topic_pages)
                print(f"📖 Topic '{topic}' -> Pages: {', '.join(map(str, topic_pages))}")
        
        sorted_pages = sorted(list(page_numbers))
        print(f"✅ Total unique pages found: {len(sorted_pages)}")
        print(f"📄 Page order: {', '.join(map(str, sorted_pages))}")
        return sorted_pages
    
    def devectorize_and_get_context(self, page_numbers: List[int], 
                                   pages_data: List[Dict[str, Any]]) -> str:
        """Get full context for specified pages"""
        print(f"\n📖 Retrieving context from {len(page_numbers)} pages...")
        
        context_parts = []
        
        for page_num in page_numbers:
            print(f"🔄 Retrieving content from page {page_num}...")
            page_data = next((p for p in pages_data if p["page_number"] == page_num), None)
            if page_data:
                context_parts.append(f"[PAGE {page_num}]\n{page_data['raw_content']}\n")
                print(f"✅ Retrieved content from page {page_num} ({len(page_data['raw_content'])} chars)")
            else:
                print(f"⚠️  Page {page_num} data not found")
        
        full_context = "\n".join(context_parts)
        print(f"📝 Total context assembled: {len(full_context)} characters from {len(context_parts)} pages")
        return full_context
    
    def take_page_screenshots(self, pdf_path: str, page_numbers: List[int], 
                             question_folder: str) -> List[str]:
        """Take screenshots of specified pages"""
        print(f"\n📸 Taking screenshots of pages: {', '.join(map(str, page_numbers))}")
        
        # Create context folder structure
        context_dir = "context"
        print(f"🔄 Creating context directory: {context_dir}")
        os.makedirs(context_dir, exist_ok=True)
        
        question_dir = os.path.join(context_dir, question_folder)
        print(f"🔄 Creating question directory: {question_dir}")
        os.makedirs(question_dir, exist_ok=True)
        
        screenshot_paths = []
        
        try:
            print("🔄 Opening PDF for screenshot extraction...")
            doc = fitz.open(pdf_path)
            
            for page_num in page_numbers:
                print(f"📸 Processing screenshot for page {page_num}...")
                if page_num <= len(doc):
                    page = doc[page_num - 1]  # PyMuPDF uses 0-based indexing
                    
                    # Get page as image
                    mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                    pix = page.get_pixmap(matrix=mat)
                    
                    # Save screenshot
                    screenshot_path = os.path.join(question_dir, f"page_{page_num}.png")
                    pix.save(screenshot_path)
                    screenshot_paths.append(screenshot_path)
                    
                    print(f"✅ Screenshot saved: {screenshot_path}")
                else:
                    print(f"⚠️  Page {page_num} exceeds document length")
            
            doc.close()
            print(f"🎉 All screenshots saved in: {question_dir}")
            
        except Exception as e:
            print(f"❌ Error taking screenshots: {e}")
        
        return screenshot_paths
    
    def generate_answer_with_references(self, question: str, context: str, 
                                       page_numbers: List[int]) -> str:
        """Generate answer with references"""
        print("\n🤖 Generating answer with references...")
        
        prompt = f"""
        Based on the following context from a PDF document, answer the question comprehensively.
        
        IMPORTANT INSTRUCTIONS:
        1. Use the information from the context to provide a detailed answer
        2. Include page references throughout your answer using [Page X] format when citing specific information
        3. Structure your answer clearly with proper formatting
        4. At the end, include a "References" section listing all pages used
        
        Question: {question}
        
        Context from PDF:
        {context}
        
        Please provide a comprehensive, well-structured answer with proper citations.
        """
        
        try:
            print("🔄 Sending request to LLM for answer generation...")
            response = self.llm.invoke(prompt)
            answer = response.content
            
            print("✅ LLM response received")
            
            # Ensure references section exists
            if "References:" not in answer and "REFERENCES:" not in answer:
                references = ", ".join([f"Page {p}" for p in sorted(page_numbers)])
                answer += f"\n\n**References:** {references}"
                print("✅ References section added to answer")
            
            print("🎉 Answer generated successfully with references")
            return answer
            
        except Exception as e:
            print(f"❌ Error generating answer: {e}")
            return f"Error generating answer: {e}"
    
    def process_document(self, pdf_path: str):
        """Process a PDF document"""
        filename = os.path.basename(pdf_path)
        print(f"\n🚀 Starting complete document processing pipeline for: {filename}")
        print("=" * 60)
        
        # Extract and analyze pages
        pages_data = self.extract_pages_with_analysis(pdf_path)
        if not pages_data:
            print("❌ No content extracted from PDF. Processing aborted.")
            return None
        
        # Create table of contents
        toc = self.create_table_of_contents(pages_data)
        
        # Vectorize content
        vectorstore = self.vectorize_content(pages_data)
        
        # Store in MongoDB
        self.store_in_mongodb(filename, pages_data, toc, vectorstore)
        
        print("\n" + "=" * 60)
        print(f"🎉 Document processing pipeline completed successfully!")
        print(f"📄 Document: {filename}")
        print(f"📊 Pages processed: {len(pages_data)}")
        print(f"📚 Topics extracted: {len(toc)}")
        print("✅ Ready for question answering!")
        print("=" * 60)
        return filename
    
    def answer_question(self, filename: str, question: str):
        """Answer a question about a processed document"""
        print(f"\n❓ Starting question answering pipeline")
        print("=" * 60)
        print(f"📄 Document: {filename}")
        print(f"❓ Question: {question}")
        print("=" * 60)
        
        # Retrieve document data from MongoDB
        print("🔄 Retrieving document data from MongoDB...")
        doc_data = self.collection.find_one({"filename": filename})
        if not doc_data:
            print(f"❌ Document not found in database: {filename}")
            print("💡 Please process the document first using --process flag")
            return
        
        print("✅ Document data retrieved from MongoDB")
        pages_data = doc_data["pages_data"]
        toc = doc_data["table_of_contents"]
        
        # Find relevant topics
        relevant_topics = self.find_relevant_topics(question, toc)
        
        # Get page numbers for topics
        relevant_pages = self.retrieve_pages_for_topics(relevant_topics, toc)
        
        if not relevant_pages:
            print("❌ No relevant pages found for the question")
            return
        
        # Get context
        context = self.devectorize_and_get_context(relevant_pages, pages_data)
        
        # Display retrieved context
        print("\n" + "="*80)
        print("📖 RETRIEVED CONTEXT FOR ANSWER GENERATION:")
        print("="*80)
        print(context[:2000] + "..." if len(context) > 2000 else context)
        print("="*80)
        
        # Take screenshots
        pdf_path = input(f"\n📸 Enter the path to {filename} for screenshots (or press Enter to skip): ").strip()
        if pdf_path and os.path.exists(pdf_path):
            question_folder = f"q_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            screenshots = self.take_page_screenshots(pdf_path, relevant_pages, question_folder)
            if screenshots:
                print(f"📸 Screenshots saved in: context/{question_folder}/")
        elif pdf_path:
            print("⚠️  PDF path not found, skipping screenshots")
        
        # Generate answer
        answer = self.generate_answer_with_references(question, context, relevant_pages)
        
        # Display final answer
        print("\n" + "="*80)
        print("🤖 GENERATED ANSWER:")
        print("="*80)
        print(answer)
        print("="*80)
        
        print(f"\n🎉 Question answering completed successfully!")
        print(f"📊 Pages referenced: {', '.join(map(str, relevant_pages))}")
        print(f"📚 Topics used: {', '.join(relevant_topics)}")

def main():
    parser = argparse.ArgumentParser(
        description="RAG CLI Application - Process PDFs and answer questions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process a document:
    python RAG_CLI.py --process /path/to/document.pdf
    
  Ask a question:
    python RAG_CLI.py --question "What is the main topic?" --document "document.pdf"
        """
    )
    parser.add_argument("--process", help="Path to PDF file to process")
    parser.add_argument("--question", help="Question to ask about processed document")
    parser.add_argument("--document", help="Document filename for question answering")
    
    args = parser.parse_args()
    
    # Validate environment variables
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY environment variable not found!")
        print("💡 Please set your Groq API key in .env file or environment")
        return
    
    try:
        # Initialize app
        print("🔧 Initializing RAG CLI Application...")
        app = RAGCLIApp()
        
        if args.process:
            if os.path.exists(args.process):
                if args.process.lower().endswith('.pdf'):
                    app.process_document(args.process)
                else:
                    print("❌ Only PDF files are supported")
            else:
                print(f"❌ File not found: {args.process}")
        
        elif args.question and args.document:
            app.answer_question(args.document, args.question)
        
        else:
            print("\n📋 RAG CLI Application")
            print("=" * 40)
            print("Usage:")
            print("  Process document:")
            print("    python RAG_CLI.py --process path/to/document.pdf")
            print("  Ask question:")
            print("    python RAG_CLI.py --question 'Your question' --document 'filename.pdf'")
            print("\n💡 Make sure to set GROQ_API_KEY and MONGODB_URI in your environment")
            
    except Exception as e:
        print(f"❌ Application error: {e}")
        print("💡 Please check your environment variables and dependencies")

if __name__ == "__main__":
    main()
