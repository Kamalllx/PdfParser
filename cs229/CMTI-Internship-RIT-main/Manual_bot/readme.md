# Manual Bot - Document Q&A Assistant

**Manual Bot** is a Streamlit-based application that enables users to interact with their PDF, text, and markdown documents using natural language queries. The system uses large language models and vector embedding techniques to deliver accurate, context-aware responses based on the document content.

## Technical Stack

- **Frontend**: Streamlit
- **Language Model**: Groq (LLaMA3-70B-8192)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Embeddings**: Hugging Face (sentence-transformers/all-mpnet-base-v2)
- **Document Parsing**: PyPDF2
- **Orchestration Framework**: LangChain

## Requirements

- Python 3.8 or higher
- Groq API key
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
    ```bash
    git clone <repository-url>
    cd Manual_bot
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the `.env` file with your API key:
    ```plaintext
    GROQ_API_KEY=your_groq_api_key_here
    ```

## Usage

1. Launch the application:
    ```bash
    streamlit run RAG.py
    ```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload your documents via the sidebar

4. Click **Process Documents** to begin indexing

5. Ask natural language questions about the uploaded content

## Document Processing Pipeline

1. **Text Extraction**
   - PDF files are parsed using PyPDF2
   - TXT and MD files are read as plain text

2. **Text Chunking**
   - Documents are split into chunks using `RecursiveCharacterTextSplitter`
   - Chunk size: 1000 characters
   - Overlap between chunks: 200 characters

3. **Embedding Generation**
   - Model: `sentence-transformers/all-mpnet-base-v2`
   - Powered by Hugging Face's Transformers library

4. **Vector Indexing and Retrieval**
   - In-memory vector search using FAISS
   - Retrieves top 4 most relevant chunks (`k=4`) for each query

5. **Conversational Query Handling**
   - Model: LLaMA3-70B-8192 via Groq API
   - Temperature: 0.2 for balanced and informative responses
   - Maintains a conversational context for follow-up questions

## Demo output for the file - ACE Maintenance manual _Fanuc-0iTF_2016.pdf
![ui](https://github.com/user-attachments/assets/210e2a6d-1f21-4b70-ae00-7acf424a5d83)
![query](https://github.com/user-attachments/assets/0c11bd8e-666a-4db1-acc3-44aae39e324f)

