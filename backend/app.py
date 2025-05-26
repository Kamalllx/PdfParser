import os
import tempfile
import uuid
from datetime import datetime
from PyPDF2 import PdfReader
import fitz  # PyMuPDF for screenshots
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pymongo
import json
import re
import shutil
from typing import List, Dict, Any

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import traceback

# Load environment variables
load_dotenv()

class RAGFlaskApp:
    def __init__(self):
        self.app = Flask(__name__)
        
        # Enable CORS for all routes
        CORS(self.app, origins=['*'])
        
        # Configuration
        self.app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
        self.app.config['UPLOAD_FOLDER'] = 'uploads'
        
        # Create necessary folders
        os.makedirs(self.app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs('stored_pdfs', exist_ok=True)  # For permanent PDF storage
        os.makedirs('context', exist_ok=True)     # For screenshots
        
        # Initialize components
        self._initialize_components()
        
        # Register routes
        self._register_routes()

    
    def _initialize_components(self):
        """Initialize embeddings, LLMs, and MongoDB connection"""
        try:
            print("🔧 Initializing components...")
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            print("✅ Embeddings model loaded")
            
            # Initialize LLMs
            self.llm = ChatGroq(
                model_name="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.2,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            
            self.vision_llm = ChatGroq(
                model_name="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.2,
                groq_api_key=os.getenv("GROQ_API_KEY")
            )
            print("✅ LLMs initialized")
            
            # MongoDB setup
            mongo_client = pymongo.MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
            self.db = mongo_client["rag_database"]
            self.collection = self.db["documents"]
            # Test connection
            self.db.list_collection_names()
            print("✅ Connected to MongoDB")
            
        except Exception as e:
            print(f"❌ Error initializing components: {e}")
            raise
    
    def _register_routes(self):
        """Register all Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
        
        @self.app.route('/documents', methods=['GET'])
        def list_documents():
            """Get list of processed documents"""
            try:
                documents = list(self.collection.find({}, {
                    'filename': 1,
                    'upload_date': 1,
                    'total_pages': 1,
                    'table_of_contents': 1,
                    '_id': 0
                }))
                
                return jsonify({
                    'success': True,
                    'documents': documents,
                    'total': len(documents)
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/process', methods=['POST'])
        def process_document():
            """Process a PDF document"""
            try:
                # Check if file is present
                if 'file' not in request.files:
                    return jsonify({
                        'success': False,
                        'error': 'No file provided'
                    }), 400
                
                file = request.files['file']
                if file.filename == '':
                    return jsonify({
                        'success': False,
                        'error': 'No file selected'
                    }), 400
                
                if not file.filename.lower().endswith('.pdf'):
                    return jsonify({
                        'success': False,
                        'error': 'Only PDF files are supported'
                    }), 400
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = os.path.join(self.app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                
                print(f"📁 File saved: {filepath}")
                
                # Process the document
                result = self._process_pdf_pipeline(filepath, filename)
                
                # Clean up uploaded file (the PDF is now stored permanently)
                os.remove(filepath)
                print(f"🧹 Temporary upload file deleted: {filepath}")
                
                return jsonify({
                    'success': True,
                    'message': 'Document processed successfully',
                    'filename': filename,
                    'processing_details': result
                })
                
            except Exception as e:
                print(f"❌ Error processing document: {e}")
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/document/<filename>/screenshots', methods=['GET'])
        def get_latest_screenshots(filename):
            """Get screenshots for a document"""
            try:
                context_dir = "context"
                if not os.path.exists(context_dir):
                    return jsonify({
                        'success': False,
                        'error': 'No screenshots available'
                    }), 404
                
                screenshots = []
                for file in os.listdir(context_dir):
                    if file.endswith('.png') and file.startswith('page_'):
                        try:
                            # Fixed the page number extraction logic
                            page_num = int(file.split('_')[1].split('.')[0])
                            screenshots.append({
                                'filename': file,
                                'page_number': page_num,
                                'path': os.path.join(context_dir, file)
                            })
                        except (ValueError, IndexError):
                            print(f"⚠️  Could not parse page number from file: {file}")
                            continue
                
                if not screenshots:
                    return jsonify({
                        'success': False,
                        'error': 'No screenshots available'
                    }), 404
                
                screenshots.sort(key=lambda x: x['page_number'])
                
                print(f"✅ Found {len(screenshots)} screenshots in context folder")
                
                return jsonify({
                    'success': True,
                    'screenshots': screenshots,
                    'total_screenshots': len(screenshots)
                })
                
            except Exception as e:
                print(f"❌ Error getting screenshots: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

        # Add a new route to serve static files from the context directory
        @self.app.route('/context/<path:filename>', methods=['GET'])
        def serve_context_file(filename):
            """Serve files from the context directory"""
            try:
                context_dir = "context"
                return send_file(os.path.join(context_dir, filename))
            except Exception as e:
                print(f"❌ Error serving file {filename}: {e}")
                return jsonify({
                    'success': False,
                    'error': f"File not found: {filename}"
                }), 404
        
        @self.app.route('/ask', methods=['POST'])
        def ask_question():
            """Ask a question about a processed document"""
            try:
                data = request.get_json()
                
                if not data or 'question' not in data or 'filename' not in data:
                    return jsonify({
                        'success': False,
                        'error': 'Question and filename are required'
                    }), 400
                
                question = data['question']
                filename = data['filename']
                
                print(f"❓ Processing question: {question}")
                print(f"📄 Document: {filename}")
                
                # Get answer
                result = self._answer_question_pipeline(filename, question)
                
                return jsonify({
                    'success': True,
                    'question': question,
                    'answer': result['answer'],
                    'context': result['context'],
                    'references': result['references'],
                    'topics_used': result['topics_used'],
                    'pages_used': result['pages_used']
                })
                
            except Exception as e:
                print(f"❌ Error answering question: {e}")
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/document/<filename>/topics', methods=['GET'])
        def get_document_topics(filename):
            """Get table of contents for a document"""
            try:
                doc_data = self.collection.find_one({"filename": filename})
                if not doc_data:
                    return jsonify({
                        'success': False,
                        'error': 'Document not found'
                    }), 404
                
                return jsonify({
                    'success': True,
                    'filename': filename,
                    'table_of_contents': doc_data['table_of_contents'],
                    'total_pages': doc_data['total_pages']
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/document/<filename>/page/<int:page_number>', methods=['GET'])
        def get_page_content(filename, page_number):
            """Get content of a specific page"""
            try:
                doc_data = self.collection.find_one({"filename": filename})
                if not doc_data:
                    return jsonify({
                        'success': False,
                        'error': 'Document not found'
                    }), 404
                
                page_data = next((p for p in doc_data['pages_data'] 
                                if p['page_number'] == page_number), None)
                
                if not page_data:
                    return jsonify({
                        'success': False,
                        'error': f'Page {page_number} not found'
                    }), 404
                
                return jsonify({
                    'success': True,
                    'page_number': page_number,
                    'content': page_data['raw_content'],
                    'topics': page_data['topics'],
                    'summary': page_data['summary'],
                    'keywords': page_data['keywords']
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    def _process_pdf_pipeline(self, pdf_path: str, original_filename: str) -> Dict[str, Any]:
        """Complete PDF processing pipeline"""
        print(f"🚀 Starting PDF processing pipeline for: {original_filename}")
        
        # Create stored_pdfs directory
        stored_pdfs_dir = "stored_pdfs"
        os.makedirs(stored_pdfs_dir, exist_ok=True)
        
        # Copy PDF to permanent storage
        stored_pdf_path = os.path.join(stored_pdfs_dir, original_filename)
        shutil.copy2(pdf_path, stored_pdf_path)
        print(f"📁 PDF stored permanently at: {stored_pdf_path}")
        
        # Extract and analyze pages
        pages_data = self._extract_pages_with_analysis(pdf_path)
        if not pages_data:
            raise Exception("No content extracted from PDF")
        
        # Create table of contents
        toc = self._create_table_of_contents(pages_data)
        
        # Vectorize content
        vectorstore = self._vectorize_content(pages_data)
        
        # Store in MongoDB (now including PDF path)
        self._store_in_mongodb(original_filename, pages_data, toc, vectorstore, stored_pdf_path)
        
        return {
            'pages_processed': len(pages_data),
            'topics_extracted': len(toc),
            'processing_timestamp': datetime.now().isoformat(),
            'pdf_stored_at': stored_pdf_path
        }

    def save_screenshots_with_cleanup(self, pdf_path: str, page_numbers: list, base_context_dir: str = "context") -> list:
        """Save screenshots with cleanup of previous images"""
        print(f"📸 Saving screenshots for pages: {', '.join(map(str, page_numbers))}")
        
        # Create context directory if it doesn't exist
        os.makedirs(base_context_dir, exist_ok=True)
        
        # Delete all previous images in the context directory
        if os.path.exists(base_context_dir):
            print("🧹 Cleaning up previous screenshots...")
            for file in os.listdir(base_context_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(base_context_dir, file)
                    os.remove(file_path)
                    print(f"🗑️  Deleted: {file}")

        screenshot_paths = []
        try:
            print("🔄 Opening PDF for screenshot extraction...")
            doc = fitz.open(pdf_path)
            
            for page_num in sorted(page_numbers):
                if page_num <= len(doc):
                    print(f"📸 Taking screenshot of page {page_num}...")
                    page = doc[page_num - 1]  # 0-based index
                    mat = fitz.Matrix(2, 2)  # zoom for better quality
                    pix = page.get_pixmap(matrix=mat)
                    screenshot_path = os.path.join(base_context_dir, f"page_{page_num}.png")
                    pix.save(screenshot_path)
                    screenshot_paths.append(screenshot_path)
                    print(f"✅ Screenshot saved: page_{page_num}.png")
                else:
                    print(f"⚠️  Page {page_num} exceeds document length")
            
            doc.close()
            print(f"🎉 All screenshots saved directly in: {base_context_dir}")
            
        except Exception as e:
            print(f"❌ Error taking screenshots: {e}")

        return screenshot_paths


    def _extract_pages_with_analysis(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF pages and analyze topics"""
        print("📖 Extracting and analyzing PDF pages...")
        
        pdf_reader = PdfReader(pdf_path)
        pages_data = []
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            print(f"🔄 Processing page {page_num}/{len(pdf_reader.pages)}")
            
            text = page.extract_text() or ""
            if not text.strip():
                continue
            
            tagged_content = f"[PAGE_{page_num}] {text}"
            topic_analysis = self._analyze_page_topics(text, page_num)
            
            page_data = {
                "page_number": page_num,
                "raw_content": text,
                "tagged_content": tagged_content,
                "topics": topic_analysis["topics"],
                "summary": topic_analysis["summary"],
                "keywords": topic_analysis["keywords"]
            }
            
            pages_data.append(page_data)
            print(f"✅ Page {page_num} processed")
        
        return pages_data
    
    def _analyze_page_topics(self, text: str, page_num: int) -> Dict[str, Any]:
        """Analyze topics and content of a page using LLM"""
        prompt = f"""
        Analyze the following text from page {page_num} and provide:
        1. Main topics/themes (list of 3-5 key topics)
        2. Brief summary (1-2 sentences)
        3. Important keywords (5-10 keywords)
        
        Text:
        {text[:2000]}
        
        Respond in this exact JSON format:
        {{
            "topics": ["topic1", "topic2", "topic3"],
            "summary": "Brief summary here",
            "keywords": ["keyword1", "keyword2", "keyword3"]
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                return analysis
            else:
                return {
                    "topics": ["General Content"],
                    "summary": f"Content from page {page_num}",
                    "keywords": ["content", "page", str(page_num)]
                }
        except Exception as e:
            print(f"⚠️ Error analyzing page {page_num}: {e}")
            return {
                "topics": ["General Content"],
                "summary": f"Content from page {page_num}",
                "keywords": ["content", "page", str(page_num)]
            }
    
    def _create_table_of_contents(self, pages_data: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Create table of contents with topics and page numbers"""
        print("📚 Creating table of contents...")
        
        toc = {}
        for page_data in pages_data:
            page_num = page_data["page_number"]
            topics = page_data["topics"]
            
            for topic in topics:
                topic_clean = topic.strip().lower()
                if topic_clean not in toc:
                    toc[topic_clean] = []
                if page_num not in toc[topic_clean]:
                    toc[topic_clean].append(page_num)
        
        # Sort page numbers for each topic
        for topic in toc:
            toc[topic].sort()
        
        print(f"✅ Created TOC with {len(toc)} topics")
        return toc
    
    def _vectorize_content(self, pages_data: List[Dict[str, Any]]) -> FAISS:
        """Vectorize tagged content"""
        print("🔢 Vectorizing content...")
        
        texts = []
        metadatas = []
        
        for page_data in pages_data:
            texts.append(page_data["tagged_content"])
            metadatas.append({
                "page_number": page_data["page_number"],
                "topics": page_data["topics"],
                "summary": page_data["summary"],
                "keywords": page_data["keywords"]
            })
        
        vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        print("✅ Vectorization complete")
        return vectorstore
    
    def _store_in_mongodb(self, filename: str, pages_data: List[Dict[str, Any]], 
                        toc: Dict[str, List[int]], vectorstore: FAISS, pdf_path: str):
        """Store all data in MongoDB"""
        print("💾 Storing data in MongoDB...")
        
        # Serialize vectorstore
        temp_dir = tempfile.mkdtemp()
        vectorstore_path = os.path.join(temp_dir, "vectorstore")
        vectorstore.save_local(vectorstore_path)
        
        with open(os.path.join(vectorstore_path, "index.faiss"), "rb") as f:
            faiss_index = f.read()
        with open(os.path.join(vectorstore_path, "index.pkl"), "rb") as f:
            faiss_metadata = f.read()
        
        document_data = {
            "filename": filename,
            "upload_date": datetime.now(),
            "pages_data": pages_data,
            "table_of_contents": toc,
            "total_pages": len(pages_data),
            "faiss_index": faiss_index,
            "faiss_metadata": faiss_metadata,
            "pdf_path": pdf_path  # Store PDF path for screenshots
        }
        
        # Store or update document
        existing = self.collection.find_one({"filename": filename})
        if existing:
            self.collection.update_one({"filename": filename}, {"$set": document_data})
            print(f"🔄 Updated existing document: {filename}")
        else:
            self.collection.insert_one(document_data)
            print(f"✅ Stored new document: {filename}")
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print("✅ Data stored in MongoDB")

    
    def _answer_question_pipeline(self, filename: str, question: str) -> Dict[str, Any]:
        """Complete question answering pipeline"""
        print(f"❓ Processing question: {question}")
        
        # Retrieve document data
        doc_data = self.collection.find_one({"filename": filename})
        if not doc_data:
            raise Exception(f"Document not found: {filename}")
        
        pages_data = doc_data["pages_data"]
        toc = doc_data["table_of_contents"]
        pdf_path = doc_data.get("pdf_path")  # Get stored PDF path
        
        # Find relevant topics
        relevant_topics = self._find_relevant_topics(question, toc)
        
        # Get page numbers for topics
        relevant_pages = self._retrieve_pages_for_topics(relevant_topics, toc)
        
        if not relevant_pages:
            raise Exception("No relevant pages found for the question")
        
        # Get context
        context = self._devectorize_and_get_context(relevant_pages, pages_data)
        
        # Generate answer
        answer = self._generate_answer_with_references(question, context, relevant_pages)
        
        # Save screenshots if PDF path is available
        screenshot_paths = []
        if pdf_path and os.path.exists(pdf_path):
            print("📸 Saving screenshots for answer context...")
            screenshot_paths = self.save_screenshots_with_cleanup(pdf_path, relevant_pages)
            print(f"✅ Screenshots saved: {len(screenshot_paths)} images")
        else:
            print("⚠️  PDF file not found, skipping screenshots")
        
        return {
            'answer': answer,
            'context': context[:1000] + "..." if len(context) > 1000 else context,
            'references': [f"Page {p}" for p in sorted(relevant_pages)],
            'topics_used': relevant_topics,
            'pages_used': relevant_pages,
            'screenshots': screenshot_paths  # Include screenshot paths in response
        }

    
    def _find_relevant_topics(self, question: str, toc: Dict[str, List[int]]) -> List[str]:
        """Find topics relevant to the question"""
        question_lower = question.lower()
        relevant_topics = []
        
        # Keyword matching
        for topic, pages in toc.items():
            if any(word in topic for word in question_lower.split() if len(word) > 2):
                relevant_topics.append(topic)
        
        # LLM fallback if no matches
        if not relevant_topics:
            topic_list = list(toc.keys())
            prompt = f"""
            Given this question: "{question}"
            And these available topics: {', '.join(topic_list)}
            
            Which topics are most relevant to answer this question?
            Return only the topic names that are relevant, separated by commas.
            """
            
            try:
                response = self.llm.invoke(prompt)
                suggested_topics = [t.strip().lower() for t in response.content.split(',')]
                relevant_topics = [t for t in suggested_topics if t in toc]
            except:
                relevant_topics = list(toc.keys())[:3]
        
        return relevant_topics or list(toc.keys())[:5]
    
    def _retrieve_pages_for_topics(self, topics: List[str], toc: Dict[str, List[int]]) -> List[int]:
        """Get ordered list of page numbers for topics"""
        page_numbers = set()
        for topic in topics:
            if topic in toc:
                page_numbers.update(toc[topic])
        return sorted(list(page_numbers))
    
    def _devectorize_and_get_context(self, page_numbers: List[int], 
                                    pages_data: List[Dict[str, Any]]) -> str:
        """Get full context for specified pages"""
        context_parts = []
        for page_num in page_numbers:
            page_data = next((p for p in pages_data if p["page_number"] == page_num), None)
            if page_data:
                context_parts.append(f"[PAGE {page_num}]\n{page_data['raw_content']}\n")
        return "\n".join(context_parts)
    
    def _generate_answer_with_references(self, question: str, context: str, 
                                       page_numbers: List[int]) -> str:
        """Generate answer with references"""
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
            response = self.llm.invoke(prompt)
            answer = response.content
            
            # Ensure references section exists
            if "References:" not in answer and "REFERENCES:" not in answer:
                references = ", ".join([f"Page {p}" for p in sorted(page_numbers)])
                answer += f"\n\n**References:** {references}"
            
            return answer
        except Exception as e:
            return f"Error generating answer: {e}"

# Create Flask app instance
def create_app():
    """Create and configure Flask application"""
    # Validate environment variables
    if not os.getenv("GROQ_API_KEY"):
        raise Exception("GROQ_API_KEY environment variable not found!")
    
    rag_app = RAGFlaskApp()
    return rag_app.app

# Create app instance
app = create_app()

if __name__ == '__main__':
    print("🚀 Starting RAG Flask Application...")
    print("📡 Server will be available at: http://localhost:5000")
    print("🔧 CORS enabled for all origins")
    print("📚 Available endpoints:")
    print("  - GET  /health - Health check")
    print("  - GET  /documents - List processed documents")
    print("  - POST /process - Process PDF document")
    print("  - POST /ask - Ask questions about documents")
    print("  - GET  /document/<filename>/topics - Get document topics")
    print("  - GET  /document/<filename>/page/<page_number> - Get page content")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
