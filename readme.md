# RAG Flask Application 

**Retrieval Augmented Generation (RAG)** system built with Flask that processes PDF documents, extracts content, creates vector embeddings, and provides intelligent question-answering capabilities with visual context through screenshots.

## **Architecture Overview**

This application implements a multi-layered RAG architecture combining document processing, vector search, and large language models to provide contextual answers from PDF documents.

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│                    (Frontend/API Clients)                       │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                     FLASK API LAYER                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐   │
│  │   Health    │ │  Document   │ │  Question   │ │ Content  │   │
│  │  Endpoints  │ │ Processing  │ │ Answering   │ │ Serving  │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                 PROCESSING LAYER                                │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐   │
│  │    PDF      │ │   Content   │ │   Vector    │ │   LLM    │   │
│  │ Extraction  │ │  Analysis   │ │ Embeddings  │ │ Analysis │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                   STORAGE LAYER                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐   │
│  │   MongoDB   │ │    FAISS    │ │    File     │ │Screenshot│   │
│  │  Document   │ │   Vector    │ │   System    │ │  Cache   │   │
│  │   Store     │ │    Store    │ │             │ │          │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘   │  
└─────────────────────────────────────────────────────────────────┘
```

## **Core Components & Libraries**

### **1. Web Framework & API Layer**
- **Flask**: Primary web framework for REST API endpoints[1]
- **Flask-CORS**: Cross-origin resource sharing for frontend integration[1]
- **Werkzeug**: WSGI utilities for secure file handling[1]

### **2. Document Processing Pipeline**
- **PyPDF2**: PDF text extraction and page-by-page content parsing[1]
- **PyMuPDF (fitz)**: High-quality PDF screenshot generation and visual rendering[1]
- **LangChain Text Splitter**: Recursive character-based text chunking[1]

### **3. Machine Learning & AI Stack**
- **HuggingFace Embeddings**: `sentence-transformers/all-mpnet-base-v2` model for semantic embeddings[1]
- **Groq ChatGroq**: Meta Llama-4 Scout model for content analysis and question answering[1]
- **LangChain Groq**: Integration layer for LLM operations[1]

### **4. Vector Database & Search**
- **FAISS (Facebook AI Similarity Search)**: High-performance vector similarity search[1]
- **LangChain Community Vectorstores**: FAISS integration for document retrieval[1]

### **5. Database & Storage**
- **PyMongo**: MongoDB client for document metadata and content storage[1]
- **MongoDB**: NoSQL database for structured document storage[1]

### **6. Utility Libraries**
- **python-dotenv**: Environment variable management[1]
- **uuid**: Unique identifier generation for file handling[1]
- **tempfile**: Temporary file management for processing[1]
- **shutil**: File system operations[1]

## **📋 Detailed Service Breakdown**

### **Service 1: Document Upload & Validation**
**Libraries**: `Flask`, `Werkzeug`, `os`

**Process**:
1. Validates file presence and PDF format[1]
2. Generates unique filenames using UUID[1]
3. Implements 50MB file size limit[1]
4. Saves to temporary upload folder[1]

### **Service 2: PDF Processing Pipeline**
**Libraries**: `PyPDF2`, `PyMuPDF`, `shutil`

**Process**:
1. **Text Extraction**: Page-by-page content extraction using PyPDF2[1]
2. **Permanent Storage**: Copies PDF to `stored_pdfs` directory[1]
3. **Content Tagging**: Adds `[PAGE_X]` tags to content for reference[1]
4. **Page Analysis**: LLM-based topic extraction per page[1]

### **Service 3: Content Analysis & Topic Extraction**
**Libraries**: `ChatGroq`, `json`, `re`

**Process**:
1. **Topic Classification**: Identifies 3-5 main topics per page[1]
2. **Summarization**: Generates 1-2 sentence summaries[1]
3. **Keyword Extraction**: Extracts 5-10 relevant keywords[1]
4. **JSON Parsing**: Structured output parsing with regex fallback[1]

### **Service 4: Vector Embedding & Indexing**
**Libraries**: `HuggingFaceEmbeddings`, `FAISS`, `LangChain`

**Process**:
1. **Text Vectorization**: Converts tagged content to embeddings[1]
2. **Metadata Association**: Links embeddings with page numbers and topics[1]
3. **FAISS Index Creation**: Builds searchable vector index[1]
4. **Serialization**: Stores index as binary data in MongoDB[1]

### **Service 5: Table of Contents Generation**
**Libraries**: Native Python collections

**Process**:
1. **Topic Aggregation**: Collects all topics across pages[1]
2. **Page Mapping**: Maps topics to page numbers[1]
3. **Sorting**: Orders page numbers for each topic[1]
4. **Deduplication**: Removes duplicate topic-page associations[1]

### **Service 6: Question Answering Pipeline**
**Libraries**: `ChatGroq`, `FAISS`, `PyMuPDF`

**Process**:
1. **Topic Matching**: Finds relevant topics using keyword matching[1]
2. **LLM Fallback**: Uses LLM when keyword matching fails[1]
3. **Page Retrieval**: Gets page numbers for relevant topics[1]
4. **Context Assembly**: Builds comprehensive context from pages[1]
5. **Answer Generation**: LLM generates cited answers[1]
6. **Screenshot Generation**: Creates visual context for referenced pages[1]

### **Service 7: Screenshot Management**
**Libraries**: `PyMuPDF`, `os`

**Process**:
1. **Cleanup**: Removes previous screenshots from context directory[1]
2. **High-Quality Rendering**: Uses 2x zoom matrix for clear images[1]
3. **Page-Specific Screenshots**: Generates images for relevant pages only[1]
4. **File Organization**: Saves as `page_X.png` in context folder[1]

## **🔄 Complete Process Flow**

### **PDF Upload & Processing Flow**

```
Upload Request
     ↓
File Validation (PDF, <50MB)
     ↓
Temporary Storage (/uploads)
     ↓
PDF Processing Pipeline
     ├── Text Extraction (PyPDF2)
     ├── Page Analysis (Groq LLM)
     ├── Topic Classification
     └── Content Tagging
     ↓
Vector Processing
     ├── Text Embedding (HuggingFace)
     ├── FAISS Index Creation
     └── Metadata Association
     ↓
Table of Contents Generation
     ├── Topic Aggregation
     ├── Page Mapping
     └── Sorting
     ↓
MongoDB Storage
     ├── Document Metadata
     ├── Page Content
     ├── Vector Index (Binary)
     └── Table of Contents
     ↓
Permanent PDF Storage (/stored_pdfs)
     ↓
Cleanup Temporary Files
     ↓
Response with Processing Details
```
### Question Answering Flow

```
Question Request
     ↓
Document Retrieval (MongoDB)
     ↓
Topic Relevance Analysis
     ├── Keyword Matching
     └── LLM Fallback
     ↓
Page Selection
     ├── Topic-to-Page Mapping
     └── Page Number Collection
     ↓
Context Assembly
     ├── Page Content Retrieval
     └── Context Concatenation
     ↓
Screenshot Generation
     ├── Previous Screenshot Cleanup
     ├── PDF Page Rendering
     └── High-Quality PNG Export
     ↓
Answer Generation (Groq LLM)
     ├── Context-Aware Processing
     ├── Citation Integration
     └── Reference Section
     ↓
Response Assembly
     ├── Answer Text
     ├── Context Summary
     ├── Page References
     └── Screenshot Paths

```

## **⚙️ Configuration & Environment**

### **Required Environment Variables**
- `GROQ_API_KEY`: Groq API authentication token[1]
- `MONGODB_URI`: MongoDB connection string (defaults to localhost)[1]

### **Application Configuration**
- **Max File Size**: 50MB[1]
- **Upload Directory**: `uploads/` (temporary)[1]
- **Storage Directory**: `stored_pdfs/` (permanent)[1]
- **Screenshot Directory**: `context/` (dynamic)[1]
- **Embedding Model**: `sentence-transformers/all-mpnet-base-v2`[1]
- **LLM Model**: `meta-llama/llama-4-scout-17b-16e-instruct`[1]

## **🚀 Deployment Architecture**

### **Directory Structure**
```
project_root/
├── app.py                 # Main application file
├── uploads/               # Temporary file storage
├── stored_pdfs/           # Permanent PDF storage
├── context/               # Dynamic screenshot storage
├── .env                   # Environment variables
└── requirements.txt       # Python dependencies
```
