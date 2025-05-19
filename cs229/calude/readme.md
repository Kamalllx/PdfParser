I'll create a CLI-based version of your RAG application with all the requested features. This will involve:

1. Converting from Streamlit to CLI
2. Adding progress tracking
3. Adding page number tagging
4. Implementing LLM analysis for topic extraction
5. Creating a table of contents
6. Adding MongoDB integration
7. Implementing topic-based retrieval
8. Processing page content by topic
9. Adding reference format for answers
10. Implementing screenshot capture
11. Using Llama-4-Scout for image processing

Let me create the complete solution for you:

I've created `RAG_CLI.py`, a comprehensive CLI-based version of your RAG application with all the requested features. Here's a breakdown of the key enhancements:

## Main Features Added

1. **CLI Interface**: Converted from Streamlit to a command-line interface using argparse
   - Process PDFs with `--pdf` argument
   - Configure MongoDB connection with `--mongo` argument

2. **Progress Tracking**:
   - Added tqdm progress bars for long operations
   - Custom spinner for CLI operations
   - Detailed color-coded status messages

3. **Page Number Tagging**:
   - Each chunk includes a page number tag in format `[PAGE X]`
   - Added topic tags to chunks `[TOPICS: topic1, topic2]`

4. **LLM Topic Analysis**:
   - Each page is analyzed using Groq's LLama-3-70b model
   - Extracts topics and summary for each page
   - Creates a comprehensive table of contents

5. **MongoDB Integration**:
   - Stores PDF metadata, including the table of contents
   - Stores each chunk with page number, topics, and vector embeddings
   - Enables efficient retrieval by topic or page

6. **Topic-Based Retrieval**:
   - Extracts topics from user queries
   - Finds relevant pages from the table of contents
   - Uses relevant pages for context

7. **Context Tracking**:
   - Prints retrieved context for transparency
   - Shows exactly what information is being used

8. **Reference Format**:
   - Answers include page references
   - Final summary shows all pages referenced

9. **Image Processing**:
   - Captures images of all PDF pages
   - Saves context images in a folder structure
   - Organizes by question ID for easy reference

10. **Vision LLM Integration**:
    - Uses Llama-4-Scout for image analysis when needed
    - OCR integration via pytesseract

## Usage Instructions

1. First, install the required dependencies:
```
pip install pymongo python-dotenv langchain langchain-groq langchain-community PyPDF2 tqdm pytesseract PyMuPDF opencv-python pillow faiss-cpu
```

2. Set up your environment variables in a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

3. Process a PDF file:
```
python RAG_CLI.py --pdf your_document.pdf
```

4. Start interacting with the document through the command-line interface by asking questions.

The application will:
- Extract text and analyze topics page by page
- Create vector embeddings with page metadata
- Store everything in MongoDB
- Generate answers with page references
- Save context images in the "context" folder

This implementation follows a more sophisticated approach than the original Streamlit version, with enhanced topic extraction, better context tracking, and comprehensive reference management.

Would you like me to explain any specific part of the implementation in more detail?
